#!/usr/bin/env python3
"""Two loopback observation hops; ServiceEvent clocks, never inferred GPU time.

Adapted from the source-pinned #98 observer (ca15a4fbfee2a6a5e85d904c902b5eb1
a8778eb16341159da0f527bb9fc3f77b). Coordinators are owned by the controller.
"""
import argparse
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import signal
import struct
import threading
import time
from urllib.parse import urlsplit

from issue103_common import append, http as json_http, new_output, read_json, require, save, stamp, vm_snapshot
from pathlib import Path


def valid_output(event, native=False):
    if native:
        return type(event.get("tokenId")) is int and event["tokenId"] >= 0
    choices = event.get("choices") or []
    delta = (choices[0].get("delta") or {}) if choices else {}
    return bool(delta.get("content")) or any(
        bool((v.get("function") or {}).get("name")) or bool((v.get("function") or {}).get("arguments"))
        for v in delta.get("tool_calls") or [])


def loopback(url):
    value = urlsplit(url)
    require(value.scheme == "http" and value.hostname == "127.0.0.1" and
            not value.username and not value.password, "Observer endpoints must be HTTP loopback")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--outer-port", type=int, required=True)
    parser.add_argument("--inner-port", type=int, required=True)
    parser.add_argument("--clone-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=int, default=3600)
    args = parser.parse_args()
    require(300 <= args.request_timeout <= 3600, "Observer timeout must be 300..3600 seconds")
    repo = Path(__file__).resolve().parents[1]
    out = new_output(args.output, repo)
    config = read_json(args.config_dir / "coordinator.json")
    worker = read_json(args.config_dir / "worker.json")
    token = Path(config["token-file"]).read_text().strip()
    registry = loopback(config["listen"])
    clone = loopback(f"http://127.0.0.1:{args.clone_port}")
    target_worker = loopback(worker["listen"])
    stop, gate, mutex = threading.Event(), threading.Lock(), threading.Lock()
    state = {"active": None, "sequence": 0, "registered": False, "fatal": None}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == "/observer/health":
                body = json.dumps({"ready": state["registered"], "fatal": state["fatal"]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.forward()

        def do_POST(self):
            self.forward()

        def body(self):
            if "chunked" not in self.headers.get("Transfer-Encoding", "").lower():
                size = int(self.headers.get("Content-Length", 0))
                require(0 <= size <= 8 * 1024 * 1024, "Observer body limit")
                return self.rfile.read(size)
            data = bytearray()
            while True:
                size = int(self.rfile.readline(128).split(b";", 1)[0].strip(), 16)
                if size == 0:
                    while self.rfile.readline(8192).strip():
                        pass
                    return bytes(data)
                require(len(data) + size <= 8 * 1024 * 1024, "Observer chunk limit")
                data.extend(self.rfile.read(size))
                require(self.rfile.read(2) == b"\r\n", "Invalid chunk delimiter")

        def forward(self):
            outer = self.server.server_port == args.outer_port
            path = self.path.split("?")[0]
            measured = outer and self.command == "POST" and path in (
                "/v1/chat/completions", "/v1/infer/stream", "/v1/messages", "/v1/responses")
            row, conn, response_sent = None, None, False
            start = time.monotonic()
            if measured:
                if not gate.acquire(timeout=args.request_timeout):
                    self.send_error(503)
                    return
            try:
                body = self.body()
                if measured:
                    before = vm_snapshot()
                    with mutex:
                        state["sequence"] += 1
                        row = {"id": f"observer-{state['sequence']:05d}", "startedUtc": stamp(),
                               "inputTokens": None, "outputTokens": None, "workerResult": None,
                               "clientTtftMs": None, "firstTokenReceived": False, "outcome": "incomplete",
                               "nativeTokenReceiptMs": [], "vmBefore": before}
                        state["active"] = row
                        append(out / "request-starts.jsonl",
                               {"id": row["id"], "startedUtc": row["startedUtc"], "vmBefore": before})
                target = clone if outer else target_worker
                conn = http.client.HTTPConnection(target.hostname, target.port, timeout=args.request_timeout)
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() in ("content-type", "authorization", "x-api-key", "x-deepinfer-token", "accept")}
                headers["Content-Length"] = str(len(body))
                start = time.monotonic()
                conn.request(self.command, self.path, body or None, headers)
                response = conn.getresponse()
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Connection", "close")
                self.end_headers()
                response_sent = True
                self.close_connection = True
                if "text/event-stream" not in response.getheader("Content-Type", ""):
                    self.wfile.write(response.read())
                    self.wfile.flush()
                else:
                    while True:
                        require(time.monotonic() - start <= args.request_timeout, "Observer total request deadline")
                        line = response.readline()
                        if not line:
                            break
                        received = time.monotonic()
                        if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                            event = json.loads(line[5:])
                            if outer and row is not None:
                                valid = valid_output(event, path == "/v1/infer/stream")
                                if path == "/v1/infer/stream" and valid:
                                    row["nativeTokenReceiptMs"].append((received - start) * 1000)
                                if event.get("error"):
                                    row["compatibleError"] = True
                                if valid and row["clientTtftMs"] is None:
                                    row["clientTtftMs"] = (received - start) * 1000
                                    row["firstTokenReceived"] = True
                            elif not outer and event.get("result"):
                                result = event["result"]
                                tokens = result.get("tokenIds")
                                with mutex:
                                    active = state["active"]
                                    if active is not None:
                                        active.update({
                                            "workerCompletedUtc": stamp(),
                                            "inputTokens": result.get("promptTokens"),
                                            "outputTokens": len(tokens) if tokens is not None else None,
                                            "workerResult": {
                                                "requestId": result.get("requestId"),
                                                "timings": result.get("timings"),
                                                "executionPath": result.get("executionPath"),
                                                "finishReason": result.get("finishReason"),
                                                "tokenIdsSha256": hashlib.sha256(b"".join(
                                                    struct.pack("<i", x) for x in tokens)).hexdigest() if tokens is not None else None,
                                            }})
                        self.wfile.write(line)
                        self.wfile.flush()
                if row is not None:
                    row["httpStatus"] = response.status
                    row["outcome"] = "completed" if (response.status == 200 and row["workerResult"] is not None
                        and row["firstTokenReceived"] and not row.get("compatibleError")) else "failed"
            except Exception as ex:
                if row is not None:
                    row["errorType"], row["outcome"] = type(ex).__name__, "failed"
                else:
                    append(out / "transport-errors.jsonl", {"utc": stamp(), "hop": "outer" if outer else "inner",
                                                           "errorType": type(ex).__name__})
                if not response_sent:
                    self.send_error(502)
            finally:
                if conn:
                    conn.close()
                if measured:
                    try:
                        if row is not None:
                            row["completedUtc"] = stamp()
                            row["observedSeconds"] = time.monotonic() - start
                            row["vmAfter"] = vm_snapshot()
                            row["newSwapOutBytes"] = row["vmAfter"]["swapOutBytes"] - row["vmBefore"]["swapOutBytes"]
                            with mutex:
                                append(out / "requests.jsonl", row)
                        else:
                            state["fatal"] = "measured request failed before resource bracket"
                            append(out / "fatal.jsonl", {"utc": stamp(), "errorType": state["fatal"]})
                    except Exception as ex:
                        state["fatal"] = type(ex).__name__
                        append(out / "fatal.jsonl", {"utc": stamp(), "errorType": type(ex).__name__})
                    finally:
                        with mutex:
                            state["active"] = None
                        gate.release()

    servers = []
    threads = []
    try:
        for port in (args.outer_port, args.inner_port):
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            server.daemon_threads = True
            servers.append(server)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            threads.append(thread)

        def heartbeat():
            while not stop.is_set():
                try:
                    code, snapshot = json_http(registry.geturl() + "/v1/workers", token)
                    require(code == 200 and len(snapshot["workers"]) == 1, "Observer registry coverage")
                    registration = snapshot["workers"][0]["registration"]
                    registration["endpoint"] = f"http://127.0.0.1:{args.inner_port}"
                    code, _ = json_http(clone.geturl() + "/v1/workers/register", token, registration)
                    require(code == 204, "Observer registration rejected")
                    state["registered"] = True
                except Exception as ex:
                    state["registered"] = False
                    append(out / "registration-errors.jsonl", {"utc": stamp(), "errorType": type(ex).__name__})
                stop.wait(1)

        heartbeat_thread = threading.Thread(target=heartbeat)
        heartbeat_thread.start()
        save(out / "manifest.json", {
            "createdUtc": stamp(), "requestTimeoutSeconds": args.request_timeout,
            "ttftBoundary": "Immediately before outer forwarding to the cloned same-arm coordinator until first valid text/tool-call output received. Pi-to-observer dispatch excluded; role/usage/keepalive excluded.",
            "workerTimingBoundary": "Unmodified terminal ServiceEvent.result.timings; no dashboard polling or clock inference.",
            "hops": 2, "promptsAndOutputTextSaved": False,
            "rawInferenceTtft": "First actual tokenId receipt, separately labeled from compatible text/tool deltas.",
            "resources": "Whole-system vm_stat immediately bracketing each outer request.",
        })
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        stop.wait()
        heartbeat_thread.join(timeout=10)
        require(not heartbeat_thread.is_alive(), "Heartbeat shutdown timeout")
    finally:
        stop.set()
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
