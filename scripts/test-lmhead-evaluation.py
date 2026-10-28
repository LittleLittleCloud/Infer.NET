#!/usr/bin/env python3
"""CPU-only checks with owned subprocesses and loopback fake HTTP, never models/GPU."""
import contextlib
import copy
import datetime as dt
import hashlib
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

import issue103_common as common

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


controller = module("issue103_controller", "run-lmhead-evaluation.py")
observer = module("issue103_observer", "observe-lmhead-e2e.py")
summarizer = module("issue103_summary", "summarize-lmhead-evaluation.py")


def observation(request_id="r1", n=100, output=11, prefill=1000, decode=500):
    return {"id": request_id, "startedUtc": "2026-10-28T00:00:01+00:00",
            "completedUtc": "2026-10-28T00:00:02+00:00", "outcome": "completed",
            "inputTokens": n, "outputTokens": output, "clientTtftMs": 12,
            "workerResult": {"requestId": request_id, "executionPath": "metal-local",
                             "timings": {"prefillMs": prefill, "decodeMs": decode, "decodeTokens": max(0, output - 1),
                                         "metalAllocatedBytes": 1}}}


def task(requests):
    return {"task_id": "HumanEval/156", "start_utc": "2026-10-28T00:00:00+00:00",
            "end_utc": "2026-10-28T00:00:03+00:00", "completed": True, "timeout": False, "error": None,
            "wall_seconds": 3,
            "pi": {"requests": [{"usage": {"input": r["inputTokens"], "output": r["outputTokens"]},
                                  "error": None} for r in requests]}}


class PowerPolicyTests(unittest.TestCase):
    def test_default_arguments_and_stable_policy_are_unchanged(self):
        argv = ["controller", "--output", "unused", "--candidate-package", "unused",
                "--candidate-sha", "a" * 40, "--candidate-source", "unused"]
        with patch.object(sys, "argv", argv):
            self.assertEqual(controller.arguments().power_policy, "stable-140w")
        with patch.object(sys, "argv", argv + ["--power-policy", "battery-assisted-35w"]):
            self.assertEqual(controller.arguments().power_policy, "battery-assisted-35w")
        self.assertEqual(common.POWER_POLICIES["stable-140w"],
                         {"adapterWatts": 140, "initialMinSoc": None, "minSoc": None, "maxSocDropPoints": 1})
        initial = {"onAC": True, "adapterWatts": 140, "soc": 50, "adapterIdentity": {"serial": "same"}}
        common.power_gate(initial)
        common.power_gate(dict(initial, soc=49), initial)
        with self.assertRaises(common.GateError):
            common.power_gate(dict(initial, soc=48), initial)

    def test_battery_assisted_initial_admission(self):
        for soc in (99, 100):
            common.power_gate({"onAC": True, "adapterWatts": 35, "soc": soc}, policy="battery-assisted-35w")
        for value in ({"onAC": True, "adapterWatts": 35, "soc": 98},
                      {"onAC": False, "adapterWatts": 35, "soc": 100},
                      {"onAC": True, "adapterWatts": 140, "soc": 100}):
            with self.subTest(value=value), self.assertRaises(common.GateError):
                common.power_gate(value, policy="battery-assisted-35w")

    def test_battery_assisted_continuous_bounds_and_adapter(self):
        for soc in (99, 100):
            initial = {"onAC": True, "adapterWatts": 35, "soc": soc, "adapterIdentity": {"serial": "same"}}
            common.power_gate(dict(initial, soc=80), initial, "battery-assisted-35w")
            for value in (dict(initial, soc=79), dict(initial, soc=soc - 21),
                          dict(initial, onAC=False), dict(initial, adapterWatts=140),
                          dict(initial, adapterIdentity={"serial": "changed"})):
                with self.subTest(initial=soc, value=value), self.assertRaises(common.GateError):
                    common.power_gate(value, initial, "battery-assisted-35w")
        self.assertEqual(common.POWER_POLICIES["battery-assisted-35w"]["maxSocDropPoints"], 20)
        initial = {"onAC": True, "adapterWatts": 35, "soc": 99}
        common.power_gate(dict(initial, soc=100), initial, "battery-assisted-35w")
        with self.assertRaises(common.GateError):
            common.power_gate(dict(initial, soc=79), initial, "battery-assisted-35w")

    def test_explicit_thermal_limits_without_false_absence_warning(self):
        for raw in ("", "Note: No thermal warning level has been recorded\n"
                    "Note: No performance warning level has been recorded\n"
                    "Note: No CPU power status has been recorded\n",
                    "CPU_Speed_Limit = 100\nCPU_Scheduler_Limit = 100",
                    "Scheduler_Limit = 100\nThermal warning level: 0",
                    "Thermal_Level = 0\nThermal Warning Level: Nominal"):
            common.thermal_gate(raw)
        for raw in ("CPU_Speed_Limit = 99", "CPU_Scheduler_Limit = 50", "Scheduler_Limit: 99.5",
                    "CPU_Speed_Limit = -1", "Thermal warning level: 1", "Thermal_Level = 2",
                    "Thermal Warning: Serious", "Thermal Warning Level = Critical"):
            for policy, watts in (("stable-140w", 140), ("battery-assisted-35w", 35)):
                with self.subTest(raw=raw, policy=policy), self.assertRaises(common.GateError):
                    common.power_gate({"onAC": True, "adapterWatts": watts, "soc": 100, "thermal": raw},
                                      policy=policy)

    def test_monitor_power_check_uses_selected_policy_and_preserves_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            c = controller.Controller.__new__(controller.Controller)
            c.out = Path(temp)
            c.args = types.SimpleNamespace(power_policy="battery-assisted-35w")
            c.initial_power = {"soc": 100, "adapterIdentity": {"serial": "same"}}
            value = {"onAC": True, "adapterWatts": 35, "soc": 80, "adapterIdentity": {"serial": "same"},
                     "thermal": "CPU_Speed_Limit = 90", "batteryRaw": "synthetic battery", "custom": "synthetic settings"}
            with patch.object(controller, "power_snapshot", return_value=value), self.assertRaises(common.GateError):
                c.check_power()
            self.assertEqual(common.rows(c.out / "power.jsonl"), [value])

    def test_preflight_plan_and_identity_bind_policy_thresholds(self):
        for policy, watts in (("stable-140w", 140), ("battery-assisted-35w", 35)):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temp:
                c = controller.Controller.__new__(controller.Controller)
                c.out = Path(temp)
                c.args = types.SimpleNamespace(
                    candidate_sha="a" * 40, candidate_source=controller.ROOT, execute=False, phase="all",
                    main_package=Path("/synthetic/main"), candidate_package=Path("/synthetic/candidate"),
                    tools=Path("/synthetic/tools"), dataset=c.out / "dataset", fixtures=c.out / "fixtures",
                    pi=Path("/synthetic/pi"), original_preflight=c.out / "original", native_driver=None,
                    native_driver_sha256=None, high_power_evidence=c.out / "ui", base_port=5380, power_policy=policy)
                c.args.dataset.write_bytes(b"synthetic")
                c.packages = {}
                c.original_hashes = {k: "synthetic" for k in ("plist", "config", "binary", "native")}
                c.original_config = {"weights": "/synthetic/weights"}
                c.original_exe, c.label = Path("/synthetic/original"), "synthetic"
                c.files = {n: c.out / f"prompt-{n}" for n in common.FIXTURES}
                health = {"pid": 42, "build": common.ORIGINAL}
                c.health = Mock(return_value=health)
                ui = {"utc": common.stamp(), "verifiedBy": "parent", "source": "synthetic UI",
                      "energyModeControls": ["High Power", "High Power"]}

                def read_json(path):
                    if path == c.args.original_preflight:
                        return {"hashes": c.original_hashes, "health": health}
                    if path == c.args.high_power_evidence:
                        return ui
                    n = int(path.parent.name)
                    return {"inputTokens": n, "textSha256": common.FIXTURES[n][0],
                            "tokenIdsSha256": common.FIXTURES[n][1]}

                def command(args, **_):
                    if args[:2] == ["git", "show"]:
                        return "\n".join(f'["{i}"] = "' + "0" * 64 + '"' for i in range(11))
                    if args[0] == "git":
                        if "status" in args:
                            return ""
                        if "merge-base" in args:
                            return common.MAIN
                        return common.TOOLS_SHA if args[2] == c.args.tools else c.args.candidate_sha
                    if args[0] == c.args.pi:
                        return "1.4.0"
                    return "pid = 42" if args[0] == "launchctl" else "{}"

                hashes = {c.args.dataset: common.DATASET}
                hashes.update({c.args.tools / "scripts" / name: digest for name, digest in common.TOOLS.items()})
                hashes.update({c.files[n]: pair[0] for n, pair in common.FIXTURES.items()})
                dataset = b"\n".join(json.dumps({"task_id": str(i)}).encode() for i in range(164))
                power = {"onAC": True, "adapterWatts": watts, "soc": 99, "thermal": "synthetic raw therm"}
                with patch.object(controller, "package", return_value={"source": "synthetic"}), \
                        patch.object(controller, "command", side_effect=command), \
                        patch.object(controller, "read_json", side_effect=read_json), \
                        patch.object(controller, "sha", side_effect=lambda p: hashes.get(Path(p), "0" * 64)), \
                        patch.object(controller.gzip, "decompress", return_value=dataset), \
                        patch.object(controller, "identity", return_value={"pid": 42}), \
                        patch.object(controller, "verify_original_identity"), \
                        patch.object(controller, "model_processes", return_value=[{"pid": 42}]), \
                        patch.object(controller, "free_ports"), \
                        patch.object(controller, "power_snapshot", return_value=power):
                    report = c.preflight()
                plan = common.read_json(c.out / "plan.json")
                self.assertEqual(json.loads(json.dumps(report)), common.read_json(c.out / "preflight.json"))
                for evidence in (report, plan, plan["identity"], c.evidence_identity):
                    self.assertEqual(evidence["powerPolicy"], policy)
                    self.assertEqual(evidence["powerThresholds"], common.POWER_POLICIES[policy])
                    self.assertEqual(bool(evidence["powerWarnings"]), policy == "battery-assisted-35w")
                self.assertEqual(plan["phaseOrder"], controller.phase_order("all", policy))
                self.assertEqual(report["supply140W"], watts == 140)
                self.assertTrue(report["powerAdmission"])
                self.assertTrue(report["freshHighPowerEvidence"])
                self.assertEqual(report["power"], power)

    def test_selected_phase_order_is_executed_once_and_recorded(self):
        for policy, expected in (("stable-140w", ["gates", "fixed", "pilot", "cohort"]),
                                 ("battery-assisted-35w", ["gates", "pilot", "cohort", "fixed"])):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temp:
                c = controller.Controller.__new__(controller.Controller)
                c.out = Path(temp)
                c.args = types.SimpleNamespace(phase="all", power_policy=policy)
                c.packages, c.evidence_identity = {}, {"powerPolicy": policy}
                c.dependency_evidence, c.pause, c.monitor, c.healthy, c.restore = [Mock() for _ in range(5)]
                called = []
                c.gates = lambda: called.append("gates")
                c.fixed = lambda: called.append("fixed")
                c.human = called.append
                c.execute()
                c.monitor_thread.join()
                self.assertEqual(called, expected)
                self.assertEqual(common.read_json(c.out / "completed.json")["phases"], expected)
                self.assertEqual([r["phase"] for r in common.rows(c.out / "phases.jsonl") if r["status"] == "starting"], expected)
                c.restore.assert_called_once()
            for phase in ("gates", "fixed", "pilot", "cohort"):
                self.assertEqual(controller.phase_order(phase, policy), [phase])

    def test_summary_carries_policy_without_inventing_results(self):
        for policy in common.POWER_POLICIES:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temp:
                run = Path(temp)
                thresholds = common.POWER_POLICIES[policy]
                common.save(run / "plan.json", {"candidate": "a" * 40, "powerPolicy": policy, "powerThresholds": thresholds})
                result = summarizer.summarize(run)
                self.assertEqual(result["powerPolicy"], policy)
                self.assertEqual(result["powerThresholds"], thresholds)
                self.assertEqual(result["powerWarnings"],
                                 [common.BATTERY_ASSISTED_WARNING] if policy == "battery-assisted-35w" else [])
                self.assertIsNone(result["formal20VersusMain"]["on"]["decodeChangePercent"])
                self.assertFalse(any(row["worker"]["complete"] for row in result["human"]))
                self.assertIsNone(result["predeclaredChecks"]["officialOnPassesNotBelowMain"])


class MetricsTests(unittest.TestCase):
    def test_pinned_standard_summarizer_agrees_on_synthetic_worker_events(self):
        source = controller.REFERENCE84 / "scripts/summarize-pi-humaneval-performance.py"
        if not source.exists():
            self.skipTest("Independent pinned checkout is unavailable")
        self.assertEqual(common.sha(source), common.TOOLS[source.name])
        official = module("pinned_standard_summary", str(source))
        observed = [observation(), observation("r2", n=200, output=21, prefill=2000, decode=2000)]
        tasks = [task(observed)]
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for entry in tasks:
                common.append(run / "tasks.jsonl", entry)
            log = run / "metrics.log"
            log.write_text("".join("INFERENCE_METRIC " + json.dumps(
                common.metric_from_observation(r, common.MAIN)) + "\n" for r in observed))
            standard = official.summarize(run, log, common.MAIN)
        checked = common.reconcile(tasks, observed, common.MAIN)
        self.assertTrue(standard["metrics_complete"])
        self.assertEqual(standard["decode_tokens_per_second"], checked["metrics"]["decodeTokensPerSecond"])
        self.assertEqual(standard["prefill_tokens_per_second"], checked["metrics"]["prefillTokensPerSecond"])

    def test_exact_reconciliation_and_n_minus_one_weighting(self):
        requests = [observation(), observation("r2", n=200, output=21, prefill=2000, decode=2000)]
        result = common.reconcile([task(requests)], requests, common.MAIN)
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["metrics"]["inputTokens"], 300)
        self.assertEqual(result["metrics"]["decodeIntervals"], 30)
        self.assertEqual(result["metrics"]["prefillTokensPerSecond"], 100)
        self.assertEqual(result["metrics"]["decodeTokensPerSecond"], 12)
        self.assertNotEqual(result["metrics"]["decodeTokensPerSecond"], (20 + 10) / 2)

    def test_missing_coverage_has_null_aggregate_not_zero(self):
        observed = [observation()]
        result = common.reconcile([task(observed * 2)], observed, common.MAIN)
        self.assertFalse(result["metrics"]["complete"])
        self.assertIsNone(result["metrics"]["decodeMs"])
        self.assertIsNone(result["metrics"]["decodeTokensPerSecond"])

    def test_token_mismatch_duplicate_and_error_rejected(self):
        observed = [observation(), observation("r2")]
        expected = task(observed)
        observed[1]["workerResult"]["requestId"] = "r1"
        observed[1]["outputTokens"] += 1
        observed[1]["outcome"] = "failed"
        result = common.reconcile([expected], observed, common.MAIN)
        self.assertTrue(any("token mismatch" in p for p in result["problems"]))
        self.assertTrue(any("duplicate" in p for p in result["problems"]))
        self.assertIsNone(result["metrics"]["prefillTokensPerSecond"])

    def test_missing_nonfinite_negative_and_zero_durations(self):
        for key in ("prefillMs", "decodeMs"):
            for value in (None, float("nan"), float("inf"), -1, 0, True):
                with self.subTest(key=key, value=value):
                    m = common.metric_from_observation(observation(), common.MAIN)
                    m[key] = value
                    result = common.aggregate([m])
                    self.assertFalse(result["complete"])
                    self.assertIsNone(result["prefillMs"])
                    json.dumps(result, allow_nan=False)

    def test_single_output_uses_zero_intervals_without_invented_decode_rate(self):
        m = common.metric_from_observation(observation(output=1, decode=0), common.MAIN)
        result = common.aggregate([m])
        self.assertTrue(result["complete"])
        self.assertEqual(result["decodeIntervals"], 0)
        self.assertIsNone(result["decodeTokensPerSecond"])

    def test_cumulative_pi_usage_is_reconciled(self):
        observed = [observation()]
        expected = task(observed)
        expected["pi"]["usage"] = {"input": 99, "output": 11}
        result = common.reconcile([expected], observed, common.MAIN)
        self.assertIn("HumanEval/156: Pi cumulative token mismatch", result["problems"])

    def test_error_task_kept_in_denominator(self):
        observed = [observation()]
        failed = task(observed)
        failed["completed"], failed["timeout"], failed["error"] = False, True, "task_timeout"
        result = common.reconcile([failed], observed, common.MAIN)
        self.assertEqual(len(result["tasks"]), 1)
        self.assertIsNone(result["metrics"]["decodeTokensPerSecond"])

    def test_worker_event_completion_avoids_observer_drain_task_boundary_error(self):
        observed = observation()
        observed["workerCompletedUtc"] = observed["completedUtc"]
        observed["completedUtc"] = "2026-10-28T00:00:04+00:00"
        self.assertTrue(common.reconcile([task([observed])], [observed], common.MAIN)["metrics"]["complete"])


class ObserverLoopbackTests(unittest.TestCase):
    def test_real_two_hop_discovery_get_and_sse_timings(self):
        token = "cpu-only-fake-token"
        registration = {"workerId": "cpu-fake", "endpoint": "unused", "capabilities": ["fake"]}
        timings = {"prefillMs": 12.5, "decodeMs": 7.5, "decodeTokens": 1}
        events = [{"heartbeat": True}, {"tokenId": 0}, {"tokenId": 42},
                  {"result": {"requestId": "cpu-request", "promptTokens": 7, "tokenIds": [0, 42],
                              "timings": timings, "executionPath": "cpu-fake", "finishReason": "stop"}}]
        stream = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        stream += b"data: [DONE]\n\n"
        calls, registrations = [], []
        lock = threading.Lock()

        class FakeHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def reply(self, status, body=b"", content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def do_GET(self):
                self.dispatch()

            def do_POST(self):
                self.dispatch()

            def dispatch(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                with lock:
                    calls.append((self.server.role, self.command, self.path,
                                  self.headers.get("X-DeepInfer-Token"), body))
                if self.headers.get("X-DeepInfer-Token") != token:
                    self.reply(403, b"{}")
                elif self.server.role == "registry" and self.path == "/v1/workers":
                    self.reply(200, json.dumps({"workers": [{"registration": registration}]}).encode())
                elif self.server.role == "clone" and self.path == "/v1/workers/register":
                    with lock:
                        registrations.append(json.loads(body))
                    self.reply(204)
                elif self.server.role == "clone":
                    with lock:
                        endpoint = registrations[-1]["endpoint"]
                    target = observer.loopback(endpoint)
                    conn = HTTPConnection(target.hostname, target.port, timeout=5)
                    try:
                        conn.request(self.command, self.path, body or None,
                                     {"Content-Type": self.headers.get("Content-Type", "application/json"),
                                      "X-DeepInfer-Token": self.headers["X-DeepInfer-Token"]})
                        response = conn.getresponse()
                        self.send_response(response.status)
                        self.send_header("Content-Type", response.getheader("Content-Type"))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        for line in iter(response.readline, b""):
                            self.wfile.write(line)
                            self.wfile.flush()
                    finally:
                        conn.close()
                elif self.server.role == "worker" and self.path == "/v1/models?cpu=1":
                    self.reply(200, b'{"data":[{"id":"cpu-fake"}]}')
                elif self.server.role == "worker" and self.path == "/v1/infer/stream":
                    self.reply(200, stream, "text/event-stream")
                else:
                    self.reply(404, b"{}")

        @contextlib.contextmanager
        def serve(role):
            server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
            server.role, server.daemon_threads = role, True
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                yield server
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())

        root = HERE.parent
        (root / "artifacts").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="issue103-observer-cpu-", dir=root / "artifacts") as temp, \
                serve("registry") as registry, serve("clone") as clone, serve("worker") as worker:
            run = Path(temp)
            (run / "token").write_text(token)
            common.save(run / "coordinator.json",
                        {"listen": f"http://127.0.0.1:{registry.server_port}", "token-file": str(run / "token")})
            common.save(run / "worker.json", {"listen": f"http://127.0.0.1:{worker.server_port}"})
            with socket.socket() as outer_socket, socket.socket() as inner_socket:
                outer_socket.bind(("127.0.0.1", 0))
                inner_socket.bind(("127.0.0.1", 0))
                outer_port, inner_port = outer_socket.getsockname()[1], inner_socket.getsockname()[1]
            # Only the OS resource sampler is synthetic; imports, HTTP and the observer are real.
            wrapper = ("import runpy, sys, issue103_common as c; "
                       "c.vm_snapshot = lambda: {'utc': c.stamp(), 'swapOutBytes': 0}; "
                       "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
            with (run / "observer.log").open("x") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-c", wrapper, str(HERE / "observe-lmhead-e2e.py"),
                     "--config-dir", str(run), "--outer-port", str(outer_port),
                     "--inner-port", str(inner_port), "--clone-port", str(clone.server_port),
                     "--output", str(run / "observer")], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
                try:
                    endpoint = f"http://127.0.0.1:{outer_port}"
                    ready = False
                    deadline = time.monotonic() + 8
                    while time.monotonic() < deadline:
                        self.assertIsNone(process.poll(), (run / "observer.log").read_text())
                        try:
                            code, status = common.http(endpoint + "/observer/health", token, timeout=.5)
                        except (common.urllib.error.URLError, TimeoutError):
                            time.sleep(.02)
                            continue
                        if code == 200 and status["ready"]:
                            self.assertIsNone(status["fatal"])
                            ready = True
                            break
                        time.sleep(.02)
                    self.assertTrue(ready, "Real observer discovery/registration did not become ready")
                    with lock:
                        self.assertEqual(registrations[0], dict(registration, endpoint=f"http://127.0.0.1:{inner_port}"))
                    conn = HTTPConnection("127.0.0.1", outer_port, timeout=5)
                    try:
                        conn.request("GET", "/v1/models?cpu=1", headers={"X-DeepInfer-Token": token})
                        response = conn.getresponse()
                        self.assertEqual(response.status, 200,
                                         common.rows(run / "observer/transport-errors.jsonl"))
                        self.assertEqual(json.loads(response.read()), {"data": [{"id": "cpu-fake"}]})
                    finally:
                        conn.close()
                    conn = HTTPConnection("127.0.0.1", outer_port, timeout=5)
                    payload = b'{"input":"cpu-only-fake","maxTokens":2}'
                    try:
                        conn.request("POST", "/v1/infer/stream", payload,
                                     {"Content-Type": "application/json", "X-DeepInfer-Token": token})
                        response = conn.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
                        self.assertEqual(response.read(), stream)
                    finally:
                        conn.close()
                    requests = []
                    deadline = time.monotonic() + 5
                    while not requests and time.monotonic() < deadline:
                        requests = common.rows(run / "observer/requests.jsonl", partial=True)
                        time.sleep(.02)
                    self.assertEqual(len(requests), 1)
                    row = requests[0]
                    self.assertEqual((row["outcome"], row["httpStatus"]), ("completed", 200))
                    self.assertEqual((row["inputTokens"], row["outputTokens"]), (7, 2))
                    self.assertEqual(row["workerResult"]["timings"], timings)
                    self.assertEqual(row["workerResult"]["requestId"], "cpu-request")
                    self.assertEqual(row["workerResult"]["tokenIdsSha256"],
                                     hashlib.sha256(struct.pack("<ii", 0, 42)).hexdigest())
                    self.assertTrue(row["firstTokenReceived"])
                    self.assertEqual(len(row["nativeTokenReceiptMs"]), 2)
                    self.assertGreater(row["clientTtftMs"], 0)
                    self.assertEqual(row["clientTtftMs"], row["nativeTokenReceiptMs"][0])
                    self.assertLessEqual(row["nativeTokenReceiptMs"][0], row["nativeTokenReceiptMs"][1])
                    self.assertLessEqual(row["nativeTokenReceiptMs"][-1], row["observedSeconds"] * 1000)
                    self.assertEqual(row["newSwapOutBytes"], 0)
                    with lock:
                        for role in ("clone", "worker"):
                            self.assertIn((role, "GET", "/v1/models?cpu=1", token, b""), calls)
                            self.assertIn((role, "POST", "/v1/infer/stream", token, payload), calls)
                    for filename in ("transport-errors.jsonl", "registration-errors.jsonl", "fatal.jsonl"):
                        self.assertFalse((run / "observer" / filename).exists(), filename)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                self.assertEqual(process.returncode, 0)


class ObserverReadinessTests(unittest.TestCase):
    def setUp(self):
        self.health = {"pid": 42, "build": "0.3.0+" + "a" * 40}
        self.registration = {
            "workerId": "issue103-on", "endpoint": "http://private.invalid:9000",
            "telemetry": {"acceptingRequests": True, **self.health,
                          "lastRequest": {"timings": {"decodeMs": 123}}},
            "devices": [{"backend": "metal", "supportsInference": True}],
            "models": [{"id": controller.MODEL, "format": "mlx-qwen36-affine-chat-v1"}],
        }
        self.snapshot = {"workers": [{"healthy": True, "registration": self.registration}]}

    def test_loading_registration_waits_for_acceptance_and_evidence_is_sanitized(self):
        self.registration["telemetry"]["acceptingRequests"] = False
        self.assertIsNone(controller.ready_registration(self.snapshot, "issue103-on", self.health))
        self.registration["telemetry"]["acceptingRequests"] = True
        evidence = controller.ready_registration(self.snapshot, "issue103-on", self.health)
        self.assertEqual(evidence, {
            "workerId": "issue103-on", "healthy": True,
            "telemetry": {"acceptingRequests": True, **self.health},
            "devices": [{"backend": "metal", "supportsInference": True}],
            "models": [{"id": controller.MODEL, "format": "mlx-qwen36-affine-chat-v1"}],
        })

    def test_ineligible_or_wrong_arm_registration_cannot_pass(self):
        for target, key, value in (
                (self.snapshot["workers"][0], "healthy", False),
                (self.registration, "workerId", "issue103-off"),
                (self.registration["telemetry"], "acceptingRequests", None),
                (self.registration["telemetry"], "pid", 43),
                (self.registration["telemetry"], "build", "0.3.0+" + "b" * 40),
                (self.registration["devices"][0], "backend", "cuda"),
                (self.registration["devices"][0], "supportsInference", False),
                (self.registration["models"][0], "id", "wrong-model"),
                (self.registration["models"][0], "format", "mlx-qwen36-affine"),
                (self.registration, "telemetry", None)):
            with self.subTest(key=key, value=value), patch.dict(target, {key: value}):
                self.assertIsNone(controller.ready_registration(self.snapshot, "issue103-on", self.health))
        for workers in ([], self.snapshot["workers"] * 2):
            with self.subTest(workers=len(workers)):
                self.assertIsNone(controller.ready_registration({"workers": workers}, "issue103-on", self.health))


class SafetyTests(unittest.TestCase):
    def test_child_start_waits_for_complete_identity_to_stabilize(self):
        with tempfile.TemporaryDirectory() as temp:
            children = common.Children(temp)
            process = Mock(pid=42)
            launcher = {"pid": 42, "comm": "/launcher/python3", "args": "/real/Python -c owned",
                        "lstart": "same"}
            stable = dict(launcher, comm="/real/Python")
            with patch.object(common.subprocess, "Popen", return_value=process), \
                    patch.object(common, "identity", side_effect=[launcher, stable, stable, stable]) as probe:
                children.start(["cpu-only"], Path(temp) / "raw.log")
            self.assertEqual(probe.call_count, 4)
            process.poll.assert_not_called()
            self.assertEqual(children.items, [(process, stable)])
            self.assertEqual(common.rows(Path(temp) / "children.jsonl")[0]["comm"], stable["comm"])
            process.poll.return_value = 0
            children.cleanup()

    def test_unstable_child_start_is_bounded_and_cleans_up_unreaped_child(self):
        with tempfile.TemporaryDirectory() as temp:
            children = common.Children(temp)
            process = Mock(pid=42)
            process.poll.return_value = None
            first = {"pid": 42, "comm": "launcher", "args": "owned", "lstart": "same"}
            with patch.object(common.subprocess, "Popen", return_value=process), \
                    patch.object(common, "identity", side_effect=[first, dict(first, comm="real")]), \
                    patch.object(common.time, "monotonic", side_effect=[0, 0, 3]), \
                    patch.object(common.time, "sleep"), \
                    self.assertRaisesRegex(common.GateError, "stabilize"):
                children.start(["cpu-only"], Path(temp) / "raw.log")
            process.terminate.assert_called_once()
            process.wait.assert_called_once()
            self.assertEqual(children.items, [])
            children.cleanup()

    def test_real_python_child_start_and_identity_checked_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            children = common.Children(temp)
            process = None
            log = Path(temp) / "python.log"
            try:
                process = children.start(
                    [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(60)"], log)
                deadline = time.monotonic() + 5
                while "ready" not in log.read_text() and time.monotonic() < deadline:
                    self.assertIsNone(process.poll())
                    time.sleep(.02)
                self.assertIn("ready", log.read_text())
                record = children.items[0][1]
                self.assertTrue(common.same_identity(record, common.identity(process.pid)))
                children.cleanup()
                self.assertIsNotNone(process.returncode)
                self.assertEqual(children.items, [])
                self.assertEqual(children.logs, [])
            finally:
                if process is not None and process.poll() is None:
                    # This Popen has not been reaped; its PID cannot be reused.
                    process.kill()
                    process.wait(timeout=5)
                children.cleanup()

    def test_live_swap_limit_excludes_only_declared_warmup(self):
        for phase in ("steady", "warmup"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temp:
                c = controller.Controller.__new__(controller.Controller)
                c.out = Path(temp)
                c.active_out = c.out / "arm"
                (c.active_out / "observer").mkdir(parents=True)
                (c.active_out / "worker.log").write_text("")
                common.append(c.active_out / "observer/request-starts.jsonl",
                              {"id": "observer-00001", "vmBefore": {"swapOutBytes": 0, "utc": common.stamp()}})
                c.observation_phases = {(str(c.active_out), "observer-00001"): phase}
                c.stop = Mock()
                c.stop.wait.side_effect = [False, True]
                c.state_lock = threading.Lock()
                c.worker, c.monitor_error = None, None
                c.worker_out = c.active_out
                c.power_last = float("inf")
                c.children = types.SimpleNamespace(items=[])
                c.check_observations = Mock()
                with patch.object(controller, "vm_snapshot", return_value={
                        "swapOutBytes": common.SWAP_LIMIT + 1, "utc": common.stamp()}):
                    c.monitor()
                if phase == "steady":
                    self.assertIsInstance(c.monitor_error, common.GateError)
                else:
                    self.assertIsNone(c.monitor_error)

    def test_short_lived_child_exit_does_not_lose_log_or_fake_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            children = common.Children(temp)
            process = Mock(pid=42, returncode=0)
            process.poll.return_value = 0
            with patch.object(common.subprocess, "Popen", return_value=process), \
                    patch.object(common, "identity", side_effect=common.GateError("already exited")):
                self.assertIs(children.start(["cpu-only"], Path(temp) / "raw.log"), process)
            children.cleanup()
            process.terminate.assert_not_called()
            self.assertEqual(common.rows(Path(temp) / "children.jsonl")[0]["earlyExitCode"], 0)

    def test_partial_live_jsonl_is_not_a_fake_resource_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            path.write_text('{"complete": 1}\n{"partial":')
            self.assertEqual(common.rows(path, partial=True), [{"complete": 1}])
            with self.assertRaises(json.JSONDecodeError):
                common.rows(path)

    def test_empty_204_registration_response_is_not_a_json_error(self):
        response = Mock(status=204)
        response.read.return_value = b""
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(common.urllib.request, "urlopen", return_value=response):
            self.assertEqual(common.http("http://127.0.0.1:1/register", "unused", {}), (204, None))

    def test_no_overwrite_and_private_output(self):
        root = HERE.parent
        (root / "artifacts").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="issue103-cpu-", dir=root / "artifacts") as temp:
            path = Path(temp) / "new"
            common.new_output(path, root)
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)
            common.save(path / "evidence.json", {"value": 1})
            with self.assertRaises(FileExistsError):
                common.new_output(path, root)
            with self.assertRaises(FileExistsError):
                common.save(path / "evidence.json", {})
            self.assertEqual(common.read_json(path / "evidence.json"), {"value": 1})

    def test_original_pid_path_and_build_are_all_required(self):
        expected = {"pid": 42, "comm": "/release/deepinfer", "args": "/release/deepinfer split-serve --config private",
                    "lstart": "same"}
        h = {"pid": 42, "build": "0.3.0+" + common.ORIGINAL}
        common.verify_original_identity(expected, Path("/release/deepinfer"), h)
        for field in ("comm", "args", "pid"):
            wrong = dict(expected)
            wrong[field] = 43 if field == "pid" else "unexpected"
            with self.assertRaises(common.GateError):
                common.verify_original_identity(wrong, Path("/release/deepinfer"), h)

    def test_reused_child_pid_is_never_signalled(self):
        process = Mock(pid=42, args=["owned"])
        process.poll.return_value = None
        expected = {"pid": 42, "comm": "python", "args": "owned", "lstart": "old"}
        with self.assertRaises(common.GateError):
            common.stop_owned(process, expected, probe=lambda _: dict(expected, lstart="reused"))
        process.send_signal.assert_not_called()
        process.kill.assert_not_called()

    def test_stable_child_executable_or_arguments_change_is_never_signalled(self):
        expected = {"pid": 42, "comm": "python", "args": "owned", "lstart": "same"}
        for field in ("comm", "args"):
            with self.subTest(field=field):
                process = Mock(pid=42, args=["owned"])
                process.poll.return_value = None
                with self.assertRaises(common.GateError):
                    common.stop_owned(process, expected, probe=lambda _: dict(expected, **{field: "changed"}))
                process.send_signal.assert_not_called()
                process.kill.assert_not_called()

    def test_reused_child_pid_is_never_signalled_during_escalation(self):
        process = Mock(pid=42, args=["owned"])
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("owned", 1)
        expected = {"pid": 42, "comm": "python", "args": "owned", "lstart": "old"}
        probe = Mock(side_effect=[expected, dict(expected, lstart="reused")])
        with self.assertRaises(common.GateError):
            common.stop_owned(process, expected, probe=probe, timeout=1)
        process.send_signal.assert_called_once_with(signal.SIGTERM)
        process.kill.assert_not_called()

    def test_bounded_owned_cleanup_escalates_only_same_identity(self):
        process = Mock(pid=42, args=["owned"])
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("owned", 1), 0]
        expected = {"pid": 42, "comm": "python", "args": "owned", "lstart": "old"}
        common.stop_owned(process, expected, probe=lambda _: expected, timeout=1)
        process.send_signal.assert_called_once_with(signal.SIGTERM)
        process.kill.assert_called_once()

    def test_pi_cleanup_gets_interrupt_for_its_own_finally(self):
        process = Mock(pid=42, args=["python", "/p/benchmark-pi-humaneval.py"])
        process.poll.return_value = None
        expected = {"pid": 42, "comm": "python", "args": "pi", "lstart": "old"}
        common.stop_owned(process, expected, probe=lambda _: expected)
        process.send_signal.assert_called_once_with(signal.SIGINT)

    def test_signal_and_exception_use_same_finally_recovery(self):
        for fail_signal in (False, True):
            with self.subTest(signal=fail_signal), tempfile.TemporaryDirectory() as temp:
                c = controller.Controller.__new__(controller.Controller)
                c.out = Path(temp)
                c.args = types.SimpleNamespace(phase="gates", power_policy="stable-140w")
                c.dependency_evidence = Mock()
                c.pause = Mock()
                c.monitor = Mock()
                c.healthy = Mock()
                c.restore = Mock()
                old_handler = signal.getsignal(signal.SIGTERM)

                def failure():
                    if fail_signal:
                        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    raise common.GateError("synthetic fail")

                c.gates = failure
                with self.assertRaises((common.GateError, InterruptedError)):
                    c.execute()
                c.restore.assert_called_once()
                self.assertEqual(signal.getsignal(signal.SIGTERM), old_handler)
                self.assertTrue((c.out / "failure.json").exists())

    def test_failed_cleanup_prevents_original_model_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            c = controller.Controller.__new__(controller.Controller)
            c.out = Path(temp)
            c.children = Mock()
            c.children.cleanup.side_effect = common.GateError("synthetic remaining model")
            c.stop = Mock()
            c.attempted_bootout = True
            c.original_health = {"ready": False}
            c.unchanged_original = Mock()
            with patch.object(controller, "wait_quiescent") as quiescent, patch.object(controller.subprocess, "run") as run:
                record = c.restore()
            quiescent.assert_not_called()
            run.assert_not_called()
            self.assertFalse(record["realRequestVerified"])
            self.assertFalse(c.restored)

    def test_ready_false_cannot_be_successful_restoration(self):
        with tempfile.TemporaryDirectory() as temp:
            c = controller.Controller.__new__(controller.Controller)
            c.out = Path(temp)
            c.children = Mock()
            c.stop = Mock()
            c.attempted_bootout = True
            c.original_health = {"ready": False}
            c.original_identity = {}
            c.original_exe = Path("/release/deepinfer")
            c.original_config = {"listen": "http://127.0.0.1:1"}
            c.token = "unused"
            c.domain, c.label, c.plist = "gui/1", "gui/1/com.infernet.metal", Path("/original.plist")
            c.unchanged_original = Mock()
            c.health = Mock(return_value={"pid": 42, "ready": False})
            times = iter((0, 0, 181))
            with patch.object(controller, "wait_quiescent"), \
                    patch.object(controller.subprocess, "run", return_value=types.SimpleNamespace(returncode=1)) as run, \
                    patch.object(controller.time, "monotonic", side_effect=lambda: next(times)), \
                    patch.object(controller.time, "sleep"), \
                    patch.object(controller, "verify_original_identity"), \
                    patch.object(controller, "identity"), \
                    patch.object(controller, "http", return_value=(503, {"ready": False})):
                record = c.restore()
            self.assertTrue(record["definitionRestored"])
            self.assertFalse(record["healthReady"])
            self.assertFalse(record["realRequestVerified"])
            self.assertFalse(c.restored)
            self.assertEqual(run.call_count, 2)

    def test_successful_restoration_requires_ready_launch_identity_and_real_request(self):
        with tempfile.TemporaryDirectory() as temp:
            c = controller.Controller.__new__(controller.Controller)
            c.out = Path(temp)
            c.children, c.stop = Mock(), Mock()
            c.attempted_bootout = True
            c.original_health, c.original_identity = {"ready": False}, {}
            c.original_exe = Path("/release/deepinfer")
            c.original_config = {"listen": "http://127.0.0.1:1"}
            c.token = "unused"
            c.domain, c.label, c.plist = "gui/1", "gui/1/com.infernet.metal", Path("/original.plist")
            c.unchanged_original = Mock()
            c.health = Mock(return_value={"pid": 42, "ready": True})
            fixture = c.out / "synthetic.txt"
            fixture.write_text("synthetic CPU fixture")
            c.files = {21: fixture}
            with patch.object(controller, "wait_quiescent"), \
                    patch.object(controller.subprocess, "run",
                                 side_effect=[types.SimpleNamespace(returncode=1), types.SimpleNamespace(returncode=0)]) as run, \
                    patch.object(controller, "verify_original_identity"), \
                    patch.object(controller, "identity", return_value={"pid": 42}), \
                    patch.object(controller, "command", return_value="pid = 42"), \
                    patch.object(controller, "http", side_effect=[(200, {"ready": True}), (200, {"tokenIds": [1]})]):
                record = c.restore()
            self.assertTrue(record["healthReady"])
            self.assertTrue(record["realRequestVerified"])
            self.assertTrue(c.restored)
            self.assertEqual(run.call_count, 2)

    def test_power_and_fresh_ui_fail_closed(self):
        baseline = {"soc": 100}
        common.power_gate({"onAC": True, "adapterWatts": 140, "soc": 99}, baseline)
        for value in ({"onAC": True, "adapterWatts": 35, "soc": 100},
                      {"onAC": False, "adapterWatts": 140, "soc": 100},
                      {"onAC": True, "adapterWatts": 140, "soc": 98}):
            with self.assertRaises(common.GateError):
                common.power_gate(value, baseline)
        now = dt.datetime.now(dt.timezone.utc)
        evidence = {"utc": now.isoformat(), "verifiedBy": "parent", "source": "synthetic UI",
                    "energyModeControls": ["High Power", "High Power"]}
        common.high_power_evidence(evidence, now)
        evidence["utc"] = (now - dt.timedelta(minutes=31)).isoformat()
        with self.assertRaises(common.GateError):
            common.high_power_evidence(evidence, now)
        with self.assertRaises(common.GateError):
            common.power_gate({"onAC": True, "adapterWatts": 140, "soc": 100, "adapterIdentity": {"serial": "changed"}},
                              {"soc": 100, "adapterIdentity": {"serial": "original"}})


class PlanObserverTests(unittest.TestCase):
    def test_fixed_medians_coverage_sequence_and_regression_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for arm in ("main", "on"):
                for n, m in common.SHAPES:
                    folder = run / f"fixed-{n}-{m}-{arm}"
                    folder.mkdir()
                    for trial in range(1, 4):
                        observed = observation(str(trial), n, m, prefill=1000 if arm == "main" else 1050,
                                               decode=500 if arm == "main" else 450)
                        observed["newSwapOutBytes"] = 0
                        response = {"tokenIds": [1] * m, "promptTokens": n, "finishReason": "length",
                                    "timings": observed["workerResult"]["timings"]}
                        common.save(folder / f"trial-{trial}.json", {
                            "excludedWarmup": False, "error": None, "inputTokens": n, "outputLimit": m,
                            "deliveredTokenIds": response["tokenIds"], "result": response,
                            "clientWallSeconds": 3, "clientTtftMs": 12})
                        common.save(folder / f"trial-{trial}-observation.json", observed)
            fixed, errors = summarizer.fixed_summary(run)
            self.assertEqual(errors, [])
            checks = summarizer.fixed_comparisons(fixed)
            self.assertTrue(all(c["prefillRegressionWithin5Percent"] for c in checks))
            self.assertTrue(all(c["decodeRegressionWithin5Percent"] for c in checks))
            broken = run / "fixed-21-128-on/trial-1.json"
            row = common.read_json(broken)
            row["result"]["tokenIds"][0] = 2
            row["deliveredTokenIds"][0] = 2
            broken.write_text(json.dumps(row))
            fixed, errors = summarizer.fixed_summary(run)
            self.assertTrue(any("exact-sequence divergence" in e for e in errors))
            affected = [r for r in fixed if r["inputTokens"] == 21]
            self.assertTrue(all(not r["worker"]["complete"] for r in affected))

    def test_missing_or_zero_denominator_never_produces_a_speedup(self):
        for value, baseline in ((None, 1), (1, None), (1, 0), (1, float("nan"))):
            self.assertIsNone(summarizer.relative_change(value, baseline))
        self.assertAlmostEqual(summarizer.relative_change(105, 100), 5)

    def test_official_negative_control_must_really_execute_assertions(self):
        for exit_code, stderr, valid in ((1, "Docker launch error", False), (125, "AssertionError", False),
                                        (1, "File /work/case.py\nAssertionError", True)):
            with self.subTest(exit=exit_code, stderr=stderr), tempfile.TemporaryDirectory() as temp:
                official = Mock()
                official.run_case.side_effect = [
                    {"status": "passed", "exit_code": 0, "stderr": ""},
                    {"status": "failed", "exit_code": exit_code, "stderr": stderr}]
                dataset = {"HumanEval/156": {"entry_point": "f", "test": "def check(f): assert f() == 1"}}
                if valid:
                    controller.official_controls(official, dataset, Path(temp))
                else:
                    with self.assertRaises(common.GateError):
                        controller.official_controls(official, dataset, Path(temp))
                self.assertTrue((Path(temp) / "official-controls.json").exists())

    def test_prior_gates_bind_harness_and_inputs_not_just_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            c = controller.Controller.__new__(controller.Controller)
            c.out = Path(temp)
            evidence = c.out / "gate.json"
            c.args = types.SimpleNamespace(phase="fixed", gates_evidence=evidence)
            c.packages = {"main": "same"}
            c.evidence_identity = {"packages": c.packages, "harnessSha256": {"controller": "new"}}
            common.save(evidence, {"passed": True, "packages": c.packages,
                                   "identity": {"packages": c.packages, "harnessSha256": {"controller": "old"}}})
            with self.assertRaises(common.GateError):
                c.dependency_evidence()

    def test_prior_gates_bind_power_policy_and_thresholds(self):
        for changed in ("powerPolicy", "powerThresholds"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                c = controller.Controller.__new__(controller.Controller)
                c.out = Path(temp)
                c.args = types.SimpleNamespace(phase="fixed", gates_evidence=c.out / "gate.json")
                c.evidence_identity = {"powerPolicy": "battery-assisted-35w",
                                       "powerThresholds": common.POWER_POLICIES["battery-assisted-35w"]}
                prior = copy.deepcopy(c.evidence_identity)
                prior[changed] = ("stable-140w" if changed == "powerPolicy"
                                  else dict(prior[changed], maxSocDropPoints=21))
                common.save(c.args.gates_evidence, {"passed": True, "identity": prior})
                with self.assertRaises(common.GateError):
                    c.dependency_evidence()

    def test_missing_error_and_nonfinite_client_durations_are_null(self):
        for value in (None, "error", -1, float("inf"), float("nan"), True):
            with self.subTest(value=value):
                summary = summarizer.describe([3, value])
                self.assertFalse(summary["complete"])
                self.assertIsNone(summary["median"])
                json.dumps(summary, allow_nan=False)
        self.assertIsNone(summarizer.describe([3], complete=False)["mean"])

    def test_main_off_on_environment_is_frozen(self):
        with patch.dict("os.environ", {"DEEPINFER_METAL_LMHEAD": "tile4", "DEEPINFER_METAL_PREFILL_GEMM": "bf16",
                                       "DYLD_LIBRARY_PATH": "/wrong", "DEEPINFER_GRAPH_FORCE_ROUTES": "/wrong"}):
            main, off, on = [controller.environment(arm) for arm in ("main", "off", "on")]
        self.assertNotIn("DEEPINFER_METAL_LMHEAD", main)
        self.assertEqual(off.pop("DEEPINFER_METAL_LMHEAD"), "off")
        self.assertEqual(on.pop("DEEPINFER_METAL_LMHEAD"), "tile4")
        self.assertEqual(main, off)
        self.assertEqual(main, on)
        self.assertNotIn("DYLD_LIBRARY_PATH", main)
        self.assertNotIn("DEEPINFER_GRAPH_FORCE_ROUTES", main)
        self.assertEqual(main["DEEPINFER_METAL_PREFILL_CHUNK"], "32")

    def test_native_selection_must_be_effective_and_off_bytes_zero(self):
        text = "LMHEAD requested=tile4 effective=tile4 layout=tile4-block256-lane32-octet8-v1 N=248320 K=2048 group=64 bits=8 extraPayloadBytes=540344320 extraAllocatedBytes=540344320 transientExtraPeakBytes=540344320 scratchBytes=0 batchedPrefill=original"
        controller.selector_evidence(text, "on")
        with self.assertRaises(common.GateError):
            controller.selector_evidence(text.replace("effective=tile4", "effective=off"), "on")
        with self.assertRaises(common.GateError):
            controller.selector_evidence(text, "main")
        with self.assertRaises(common.GateError):
            controller.selector_evidence(text.replace("tile4", "off").replace("extraPayloadBytes=540344320", "extraPayloadBytes=0"), "off")

    def test_role_heartbeat_and_usage_are_not_first_tokens(self):
        for event in ({}, {"choices": [{"delta": {"role": "assistant"}}]}, {"usage": {"output": 1}}, {"tokenId": -1}):
            self.assertFalse(observer.valid_output(event))
            self.assertFalse(observer.valid_output(event, native=True))
        self.assertTrue(observer.valid_output({"choices": [{"delta": {"content": "x"}}]}))
        self.assertTrue(observer.valid_output({"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "{"}}]}}]}))
        self.assertTrue(observer.valid_output({"tokenId": 0}, native=True))

    def test_no_logit_state_claim_or_fabricated_missing_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            common.save(run / "plan.json", {"candidate": "a" * 40})
            result = summarizer.summarize(run)
            self.assertEqual(len(result["fixed"]), 10)
            self.assertEqual(result["formal20VersusMain"]["on"]["decodeChangePercent"], None)
            self.assertTrue(all(not r["worker"]["complete"] for r in result["human"]))
            self.assertTrue(any("logits" in item for item in result["limitations"]))
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
