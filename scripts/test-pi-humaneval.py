"""Offline fixtures only: never start Pi or contact an inference endpoint."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).with_name("benchmark-pi-humaneval.py")
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def fixture():
    message = {
        "role": "assistant", "provider": "infernet", "model": benchmark.MODEL,
        "usage": {"input": 10, "output": 3, "cacheRead": 2, "totalTokens": 15},
        "stopReason": "stop", "durationMs": 1200,
    }
    return [
        {"seconds": 0, "event": {"type": "turn_start"}},
        {"seconds": .1, "event": {"type": "message_start", "message": {"role": "assistant"}}},
        {"seconds": .2, "event": {"type": "message_update", "usage": message["usage"]}},
        {"seconds": 1.3, "event": {"type": "message_end", "message": message}},
        {"seconds": 1.4, "event": {
            "type": "tool_execution_start", "toolCallId": "a", "toolName": "write",
            "args": {"path": "solution.py", "content": "def f(): pass\n"},
        }},
        {"seconds": 1.5, "event": {
            "type": "tool_execution_end", "toolCallId": "a", "toolName": "write",
            "isError": False, "durationMs": 100,
        }},
        {"seconds": 1.6, "event": {"type": "turn_end", "message": message}},
        {"seconds": 1.7, "event": {"type": "agent_end", "messages": [message]}},
        {"seconds": 1.8, "event": {"type": "agent_settled", "aborted": False}},
    ]


class BenchmarkTests(unittest.TestCase):
    def test_usage_is_not_double_counted(self):
        parsed = benchmark.parse_events(fixture())
        self.assertEqual(parsed["usage"]["totalTokens"], 15)
        self.assertEqual(len(parsed["requests"]), 1)
        self.assertAlmostEqual(parsed["requests"][0]["observed_seconds"], 1.2)
        self.assertEqual(parsed["requests"][0]["duration_ms"], 1200)
        self.assertEqual(parsed["turn_seconds"], [1.6])
        self.assertEqual(len(parsed["tool_calls"]), 2)
        self.assertTrue(parsed["agent_ok"])

    def test_failures_stay_in_denominator(self):
        rows = [
            {"wall_seconds": seconds, "completed": ok, "syntax_status": syntax,
             "timeout": timeout, "error": error, "pi": benchmark.parse_events(events)}
            for seconds, ok, syntax, timeout, error, events in [
                (2, True, "ok", False, None, fixture()),
                (10, False, "missing", True, "task_timeout", []),
                (3, False, "invalid", False, "failed", []),
            ]
        ]
        summary = benchmark.summarize(rows)
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["completion_count"], 1)
        self.assertEqual(summary["syntax_ok_count"], 1)
        self.assertEqual(summary["timeout_count"], 1)
        self.assertEqual(summary["error_count"], 2)
        self.assertEqual(summary["wall_seconds"], {"median": 3, "p90": 10, "total": 15})
        self.assertEqual(summary["tokens"]["totalTokens"], 15)
        self.assertFalse(summary["correctness_evaluated"])
        self.assertIsNone(benchmark.summarize([])["wall_seconds"]["p90"])

    def test_malformed_truncated_and_failed_streams(self):
        for reason in ("error", "aborted", "length", "toolUse"):
            events = fixture()
            events[3]["event"]["message"]["stopReason"] = reason
            self.assertFalse(benchmark.parse_events(events)["agent_ok"])
        self.assertFalse(benchmark.parse_events(fixture()[:4])["agent_ok"])
        self.assertFalse(benchmark.parse_events(fixture() + [{"seconds": 2, "event": None}])["agent_ok"])
        self.assertFalse(benchmark.parse_events([{"seconds": 0, "event": []}])["agent_ok"])
        events = fixture()
        events[3]["event"]["message"]["provider"] = "another-provider"
        self.assertTrue(benchmark.parse_events(events)["model_mismatch"])

    def test_two_requests_and_aborted_settlement(self):
        events = fixture() + [
            {"seconds": 2, "event": {"type": "turn_start"}},
            {"seconds": 2.1, "event": {
                "type": "message_start", "message": {"role": "assistant"},
            }},
            {"seconds": 3, "event": fixture()[3]["event"]},
            {"seconds": 3.1, "event": {"type": "turn_end"}},
        ]
        parsed = benchmark.parse_events(events)
        self.assertEqual(parsed["usage"]["input"], 20)
        self.assertEqual(len(parsed["requests"]), 2)
        events.append({"seconds": 4, "event": {"type": "agent_settled", "aborted": True}})
        self.assertFalse(benchmark.parse_events(events)["agent_ok"])

    def test_selection_is_stable_and_nested(self):
        tasks = {f"HumanEval/{i}": "" for i in range(164)}
        pilot = benchmark.select_ids(tasks, "pilot", 0)
        cohort = benchmark.select_ids(tasks, "cohort", 0)
        self.assertEqual(len(pilot), 5)
        self.assertEqual(len(cohort), 20)
        self.assertEqual(pilot, cohort[:5])
        self.assertEqual(cohort, benchmark.select_ids(dict(reversed(list(tasks.items()))), "cohort", 0))
        self.assertNotEqual(pilot, benchmark.select_ids(tasks, "pilot", 1))
        self.assertEqual(len(benchmark.select_ids(tasks, "full", 0)), 164)
        self.assertEqual(benchmark.select_ids(tasks, "pilot", 0, "HumanEval/2"), ["HumanEval/2"])
        for ids in ("HumanEval/2,HumanEval/2", "../test", ""):
            with self.assertRaises(ValueError):
                benchmark.select_ids(tasks, "pilot", 0, ids)

    def test_node_entrypoint_preserves_prompt_as_one_argument(self):
        prompt = 'def f():\n    """A & B | C %PATH%."""\n'
        command = benchmark.pi_command(["node", "C:\\Pi Space\\cli.js"], prompt)
        self.assertEqual(command[:2], ["node", "C:\\Pi Space\\cli.js"])
        self.assertEqual(command[-2:], ["-p", prompt])

    def test_dataset_pin_rejects_other_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.gz"
            path.write_bytes(b"not the pinned dataset")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                benchmark.load_dataset(path)

    def test_capture_redacts_decoded_json_and_stderr(self):
        token = 'fixture-only-"quoted-credential'
        event = {"type": "session", "text": token + "\u2028still one record"}
        program = (
            "import sys; print(" + repr(json.dumps(event)) + "); "
            "print(" + repr(token) + ", file=sys.stderr)"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            result = benchmark.capture(
                [sys.executable, "-c", program], temporary, os.environ.copy(), 5, token, path
            )
            self.assertEqual(result["returncode"], 0)
            records = [json.loads(line) for line in path.read_text().split("\n") if line]
            stdout = [r for r in records if "event" in r]
            self.assertEqual(len(stdout), 1)
            self.assertIn("[REDACTED]", stdout[0]["event"]["text"])
            self.assertNotIn(token, json.dumps(records))
            self.assertEqual(result["pi"]["malformed_records"], 0)

    def test_timeout_and_launch_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            result = benchmark.capture(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                temporary, os.environ.copy(), .15, "", path,
            )
            self.assertTrue(result["timeout"])
            self.assertEqual(result["error"], "task_timeout")
            self.assertLess(result["wall_seconds"], 5)
            result = benchmark.capture(
                [str(Path(temporary) / "missing")], temporary, {}, 1, "", path
            )
            self.assertEqual(result["error"], "FileNotFoundError")

    def test_task_has_fresh_config_and_only_prompt(self):
        token = "fixture-only-private-value"
        workspaces = []
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(
                output=Path(temporary), endpoint="http://localhost:5200",
                timeout=5, context_window=32768, max_tokens=4096,
                endpoint_build=None, model_revision=None,
            )

            def fake_capture(command, workspace, env, timeout, secret, event_path):
                workspaces.append(str(workspace))
                self.assertEqual(list(workspace.iterdir()), [])
                self.assertIn("--no-session", command)
                self.assertIn("read,write,bash,edit", command)
                self.assertNotIn("--no-tools", command)
                self.assertIn("def f():", command[-1])
                self.assertNotIn("canonical_solution", command[-1])
                config = (Path(env["PI_CODING_AGENT_DIR"]) / "models.json").read_text()
                self.assertNotIn(token, config)
                provider = json.loads(config)["providers"]["infernet"]
                self.assertEqual(provider["apiKey"], "$INFERNET_BENCH_TOKEN")
                self.assertTrue(provider["authHeader"])
                self.assertNotIn("UNRELATED_SECRET", env)
                (workspace / "solution.py").write_text("def f():\n    return 1\n")
                return {
                    "returncode": 0, "error": None, "timeout": False,
                    "wall_seconds": 2, "pi": benchmark.parse_events(fixture()),
                }

            with patch.object(benchmark, "capture", side_effect=fake_capture):
                rows = [
                    benchmark.run_task(args, f"HumanEval/{i}", "def f():\n", "pi", "fixture", token)
                    for i in range(2)
                ]
            self.assertNotEqual(workspaces[0], workspaces[1])
            self.assertTrue(all(r["completed"] and r["syntax_status"] == "ok" for r in rows))
            self.assertEqual(rows[0]["solution_sha256"], rows[1]["solution_sha256"])
            self.assertTrue(all(not Path(w).exists() for w in workspaces))

    def test_main_retains_unexpected_failure_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            token_file, output = root / "token", root / "output"
            token_file.write_text("fixture-only-private-value")
            argv = [
                "benchmark", "--token-file", str(token_file), "--output", str(output),
                "--ids", "HumanEval/0,HumanEval/1",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(benchmark, "load_dataset", return_value={
                    "HumanEval/0": "def f():\n", "HumanEval/1": "def g():\n",
                }),
                patch.object(benchmark.shutil, "which", return_value=sys.executable),
                patch.object(benchmark.subprocess, "check_output", return_value=b"fixture"),
                patch.object(benchmark, "run_task", side_effect=OSError("private error text")),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(benchmark.main(), 1)
                rows = [json.loads(line) for line in (output / "tasks.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(r["error"] == "OSError" for r in rows))
                self.assertTrue(all(r["command"] and r["prompt_sha256"] for r in rows))
                self.assertEqual(json.loads((output / "summary.json").read_text())["count"], 2)
                with self.assertRaises(SystemExit) as raised:
                    benchmark.main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
