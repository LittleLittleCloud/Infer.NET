#!/usr/bin/env python3
"""CPU-only compatible API and optional Pi 1.4.0 tests with a loopback fake worker."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen3.6-35b-a3b"
TOKEN = "cpu-only-chat-termination-not-a-credential"
TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
VALID = "<tool_call><function=read><parameter=path>fixture.txt</parameter></function></tool_call>"
# Reproduces the exception class/accounting failure, not the unretained private text.
MALFORMED = VALID.removesuffix("</tool_call>")


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Worker(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            data = bytearray()
            while True:
                size = int(self.rfile.readline().strip(), 16)
                if size == 0:
                    self.rfile.readline()
                    break
                data.extend(self.rfile.read(size))
                self.rfile.read(2)
        else:
            data = self.rfile.read(int(self.headers["Content-Length"]))
        request = json.loads(data)
        self.server.requests.append(request)
        mode = next(m["content"].removeprefix("cpu-case:") for m in request["messages"]
                    if m["content"].startswith("cpu-case:"))
        prompt, ids, text, finish = 23, [1, 2], "synthetic", "eos"
        if mode == "malformed":
            prompt, ids, text = 3645, [1] * 151, MALFORMED
        elif mode == "valid-tool" and "<tool_response>" not in request["messages"][-1]["content"]:
            text = VALID
        elif mode == "empty":
            ids, text = [], ""
        elif mode == "length":
            finish = "length"
        elif mode == "unknown-finish":
            finish = "unknown"
        elif mode == "null-text":
            text = None
        result = dict(requestId="cpu", workerId="cpu", model=MODEL, backend="metal",
                      text=text, tokens=[], isSimulation=True, computeMs=1,
                      tokenIds=ids, promptTokens=prompt, finishReason=finish)
        self.send_response(200)
        streaming = self.path.endswith("/stream")
        self.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        self.end_headers()
        if not streaming:
            if mode == "missing-ids":
                result["tokenIds"] = None
            self.wfile.write(json.dumps(result).encode())
            return
        for index, token_id in enumerate(ids):
            self.event(dict(text="synthetic" if index == 0 else "", tokenId=token_id, promptTokens=prompt))
        if mode == "truncate":
            return
        if mode == "fault":
            self.event(dict(error={"code": "cpu_fault", "message": "Synthetic worker failure."}))
            return
        if mode == "count-mismatch":
            result["tokenIds"] = [1]
        if mode == "prompt-mismatch":
            result["promptTokens"] += 1
        if mode == "missing-ids":
            result["tokenIds"] = None
        self.event(dict(result=result))

    def event(self, value):
        self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())
        self.wfile.flush()


class TerminationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="chat-termination-", dir=ROOT / "artifacts")
        cls.directory = Path(cls.temp.name)
        cls.worker = ThreadingHTTPServer(("127.0.0.1", 0), Worker)
        cls.worker.requests = []
        cls.thread = threading.Thread(target=cls.worker.serve_forever)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{port()}"
        cls.log = (cls.directory / "host.log").open("w")
        cls.process = subprocess.Popen(
            [*HOST, "coordinator", "--listen", cls.endpoint, "--token", TOKEN],
            stdout=cls.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while True:
                if cls.process.poll() is not None:
                    raise RuntimeError("CPU host exited: " + (cls.directory / "host.log").read_text())
                try:
                    if cls.call("/health")[0] == 200:
                        break
                except urllib.error.URLError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.05)
            registration = dict(
                protocolVersion=1, workerId="cpu", endpoint=f"http://127.0.0.1:{cls.worker.server_port}",
                devices=[dict(id="cpu-fake", backend="metal", name="CPU fake", architecture="fake",
                              memoryBytes=None, nativeAvailable=False, supportsInference=True, status="synthetic")],
                models=[dict(id=MODEL, format="cpu-fake-chat-v1", isSimulation=True)], capacity=1)
            assert cls.call("/v1/workers/register", registration)[0] == 204
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.log.close()
        cls.worker.shutdown()
        cls.worker.server_close()
        cls.thread.join(timeout=5)
        cls.temp.cleanup()

    @classmethod
    def call(cls, path, body=None):
        request = urllib.request.Request(cls.endpoint + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={"X-DeepInfer-Token": TOKEN, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def request(self, mode, stream=True, path="/v1/chat/completions", usage=True, tools=False):
        body = {"model": MODEL, "stream": stream,
                "input" if path.endswith("responses") else "messages": [{"role": "user", "content": "cpu-case:" + mode}],
                "max_output_tokens" if path.endswith("responses") else "max_tokens": 4096}
        if path.endswith("completions"):
            if stream:
                body["stream_options"] = {"include_usage": usage}
            if tools:
                body["tools"] = TOOLS
        status, text = self.call(path, body)
        events = [json.loads(line[6:]) for line in text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"] if stream else [json.loads(text)]
        return status, text, events

    def test_invalid_tool_is_an_accounted_error_not_a_completion(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                status, text, events = self.request("malformed", stream, tools=True)
                self.assertEqual(status, 200 if stream else 502)
                self.assertEqual(events[-1]["error"]["type"], "api_error")
                self.assertEqual(events[-1]["error"]["code"], "invalid_model_output")
                usage = events[-2]["usage"] if stream else events[-1]["usage"]
                self.assertEqual(usage, dict(prompt_tokens=3645, completion_tokens=151, total_tokens=3796))
                self.assertNotIn("[DONE]", text)
                self.assertNotIn("fixture.txt", text)
                for event in events:
                    for choice in event.get("choices", []):
                        self.assertIsNone(choice["finish_reason"])
                        self.assertNotIn("tool_calls", choice["delta"])

    def test_no_requested_usage_does_not_add_usage_only_chunks(self):
        _, _, events = self.request("malformed", usage=False, tools=True)
        self.assertFalse(any("usage" in e for e in events[:-1]))
        self.assertEqual(events[-1]["usage"]["completion_tokens"], 151)

    def test_valid_tools_and_text_keep_real_usage_and_finish(self):
        for mode, finish in (("normal", "stop"), ("length", "length"), ("empty", "stop"), ("valid-tool", "tool_calls")):
            for stream in (False, True):
                with self.subTest(mode=mode, stream=stream):
                    status, text, events = self.request(mode, stream, tools=mode == "valid-tool")
                    self.assertEqual(status, 200)
                    terminal = events[-2] if stream else events[0]
                    self.assertEqual(terminal["choices"][0]["finish_reason"], finish)
                    self.assertEqual(events[-1]["usage"]["completion_tokens"], 0 if mode == "empty" else 2)
                    self.assertEqual(events[-1]["usage"]["prompt_tokens"], 23)
                    if stream:
                        self.assertTrue(text.endswith("data: [DONE]\n\n"))

    def test_failed_or_unaccounted_upstream_never_gets_success_termination(self):
        for path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
            for mode in ("truncate", "fault", "count-mismatch", "prompt-mismatch", "missing-ids", "unknown-finish", "null-text"):
                with self.subTest(path=path, mode=mode):
                    _, text, events = self.request(mode, path=path)
                    self.assertNotIn("[DONE]", text)
                    self.assertNotIn("response.completed", text)
                    self.assertNotIn("message_stop", text)
                    self.assertTrue(any("error" in e or e.get("type") == "response.failed" for e in events))
                    if mode in ("truncate", "fault"):
                        error = events[-1].get("response", events[-1])
                        self.assertIsNone(error.get("usage"))
                        self.assertEqual(error["observed_usage"],
                                         dict(input_tokens=23, output_tokens_at_least=2, complete=False))

    def test_nonstream_invalid_metadata_is_not_length_success(self):
        for path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
            for mode in ("unknown-finish", "missing-ids", "null-text"):
                with self.subTest(path=path, mode=mode):
                    status, _, events = self.request(mode, False, path)
                    self.assertEqual(status, 502)
                    self.assertIn("error", events[0])

    def test_pi_consumes_genuine_termination_and_usage(self):
        if not PI:
            self.skipTest("Pass --pi to exercise the installed Pi 1.4.0 client")
        self.assertEqual(subprocess.check_output([PI, "--version"], text=True).strip(), "1.4.0")
        for mode in ("normal", "length", "empty", "malformed", "valid-tool", "truncate", "fault"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=self.directory) as temporary:
                directory = Path(temporary)
                agent = directory / "agent"
                agent.mkdir()
                (directory / "fixture.txt").write_text("synthetic fixture\n")
                config = {"providers": {"infernet-cpu": {
                    "baseUrl": self.endpoint + "/v1", "api": "openai-completions",
                    "apiKey": "$CPU_TEST_TOKEN", "authHeader": True, "models": [{
                        "id": MODEL, "reasoning": False, "input": ["text"], "contextWindow": 32768, "maxTokens": 4096,
                        "compat": {"maxTokensField": "max_tokens", "supportsDeveloperRole": False,
                                   "supportsReasoningEffort": False, "supportsStore": False}}]}}}
                (agent / "models.json").write_text(json.dumps(config))
                (agent / "settings.json").write_text(json.dumps({"compaction": {"enabled": False}, "retry": {"enabled": False}}))
                env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL")}
                env.update(HOME=str(directory), TMPDIR=str(directory), PI_CODING_AGENT_DIR=str(agent),
                           CPU_TEST_TOKEN=TOKEN, PI_OFFLINE="1")
                before = len(self.worker.requests)
                run = subprocess.run(
                    [PI, "--provider", "infernet-cpu", "--model", MODEL, "--mode", "json", "--no-session",
                     "--tools", "read", "--no-extensions", "--no-mcp", "--no-skills", "--no-prompt-templates",
                     "--no-themes", "--no-context-files", "--no-approve", "--thinking", "off", "-p", "cpu-case:" + mode],
                    cwd=directory, env=env, capture_output=True, text=True, timeout=30)
                events = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
                messages = [e["message"] for e in events if e.get("type") == "message_end" and
                            e.get("message", {}).get("role") == "assistant"]
                self.assertTrue(messages, run.stderr)
                message = messages[0]
                if mode in ("malformed", "truncate", "fault"):
                    self.assertEqual(message["stopReason"], "error")
                    if mode == "malformed":
                        self.assertIn("Invalid model output", message["errorMessage"])
                        self.assertEqual((message["usage"]["input"], message["usage"]["output"]), (3645, 151))
                    self.assertFalse(any(e.get("type") == "tool_execution_start" for e in events))
                    self.assertEqual(len(self.worker.requests), before + 1)
                else:
                    self.assertEqual(message["stopReason"], "length" if mode == "length" else
                                     "toolUse" if mode == "valid-tool" else "stop")
                    self.assertEqual((message["usage"]["input"], message["usage"]["output"]), (23, 0 if mode == "empty" else 2))
                    if mode == "valid-tool":
                        self.assertEqual(len(messages), 2)
                        self.assertTrue(any(e.get("type") == "tool_execution_start" for e in events))
                        self.assertEqual(messages[-1]["stopReason"], "stop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable")
    parser.add_argument("--pi")
    args, remaining = parser.parse_known_args()
    HOST = [args.executable] if args.executable else ["dotnet", str(ROOT / "src/DeepInfer.Host/bin/Release/net10.0/deepinfer.dll")]
    PI = args.pi
    (ROOT / "artifacts").mkdir(exist_ok=True)
    unittest.main(argv=[__file__, *remaining])
