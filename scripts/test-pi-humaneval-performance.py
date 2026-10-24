#!/usr/bin/env python3
"""Offline checks for the HumanEval worker-performance rollup."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("summarize-pi-humaneval-performance.py")
SPEC = importlib.util.spec_from_file_location("humaneval_performance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PerformanceSummaryTests(unittest.TestCase):
    def test_aggregates_worker_timings_and_detects_missing_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps({
                "dataset_sha256": "dataset", "ids": ["HumanEval/1"],
                "pi_version": "1.4.0", "model": "model", "endpoint_build": "build",
            }))
            (run / "summary.json").write_text(json.dumps({
                "request_count": 2, "wall_seconds": {"total": 8.0},
                "completion_count": 1, "count": 1,
            }))
            (run / "tasks.jsonl").write_text(json.dumps({
                "start_utc": "2026-10-23T00:00:00+00:00",
                "end_utc": "2026-10-23T00:00:10+00:00",
            }) + "\n")
            log = root / "serve.log"
            log.write_text(
                "INFERENCE_METRIC request=a completed=2026-10-23T00:00:02.1234567+00:00 build=abc input=100 output=11 prefillMs=50 decodeTokens=10 decodeMs=100 path=cuda-metal-split\n"
                "INFERENCE_METRIC request=b completed=2026-10-23T00:00:04+00:00 build=abc input=300 output=21 prefillMs=150 decodeTokens=20 decodeMs=200 path=cuda-metal-split\n"
                "INFERENCE_METRIC request=c completed=2026-10-23T00:00:20+00:00 build=abc input=999 output=2 prefillMs=1 decodeTokens=1 decodeMs=1 path=local\n"
            )
            sha = "a" * 40
            result = MODULE.summarize(run, [log], sha, sha, "GEMM adjustment")
            self.assertTrue(result["metrics_complete"])
            self.assertEqual(result["request_metrics_count"], 2)
            self.assertEqual(result["prefill_tokens_per_second"], 2000)
            self.assertEqual(result["decode_tokens_per_second"], 100)
            self.assertEqual(result["change"], "GEMM adjustment")
            log.write_text(log.read_text().splitlines()[0] + "\n")
            partial = MODULE.summarize(run, [log], sha, sha, "GEMM adjustment")
            self.assertFalse(partial["metrics_complete"])
            self.assertEqual(partial["request_metrics_count"], 1)


if __name__ == "__main__":
    unittest.main()
