#!/usr/bin/env python3
"""Aggregate per-request Infer.NET worker timings for one Pi HumanEval run."""

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics


MARKER = "INFERENCE_METRIC "
SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
FIELDS = ("request", "completed", "build", "input", "output", "prefillMs", "decodeTokens", "decodeMs", "path")


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_metric(line):
    if MARKER not in line:
        return None
    fields = dict(part.split("=", 1) for part in line.split(MARKER, 1)[1].strip().split() if "=" in part)
    if any(key not in fields for key in FIELDS):
        raise ValueError("Incomplete INFERENCE_METRIC record")
    row = {key: fields[key] for key in FIELDS}
    row["completed"] = timestamp(row["completed"])
    for key in ("input", "output", "decodeTokens"):
        row[key] = int(row[key])
    for key in ("prefillMs", "decodeMs"):
        row[key] = float(row[key])
    if (any(row[key] < 0 for key in ("input", "output", "decodeTokens", "prefillMs", "decodeMs"))
            or not math.isfinite(row["prefillMs"]) or not math.isfinite(row["decodeMs"])):
        raise ValueError("Invalid INFERENCE_METRIC measurement")
    return row


def summarize(run, logs, mac_commit, windows_commit, change):
    tasks = [json.loads(line) for line in (run / "tasks.jsonl").read_text().splitlines() if line]
    manifest = json.loads((run / "manifest.json").read_text())
    pi = json.loads((run / "summary.json").read_text())
    if not tasks:
        raise ValueError("No HumanEval tasks")
    start = min(timestamp(task["start_utc"]) for task in tasks)
    end = max(timestamp(task["end_utc"]) for task in tasks)
    records = {}
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            metric = parse_metric(line)
            if metric is not None and start <= metric["completed"] <= end:
                if metric["request"] in records:
                    raise ValueError("Duplicate worker request metric")
                records[metric["request"]] = metric
    rows = list(records.values())
    prefill_ms = sum(row["prefillMs"] for row in rows)
    decode_ms = sum(row["decodeMs"] for row in rows)
    decode_tokens = sum(row["decodeTokens"] for row in rows)
    expected = pi["request_count"]
    complete = len(rows) == expected and expected > 0
    return {
        "run": str(run.resolve()),
        "dataset_sha256": manifest["dataset_sha256"],
        "task_ids": manifest["ids"],
        "pi_version": manifest["pi_version"],
        "model": manifest["model"],
        "endpoint_build": manifest.get("endpoint_build"),
        "mac_commit": mac_commit,
        "windows_commit": windows_commit,
        "change": change,
        "request_count_expected": expected,
        "request_metrics_count": len(rows),
        "metrics_complete": complete,
        "worker_builds": sorted({row["build"] for row in rows}),
        "execution_paths": {path: sum(row["path"] == path for row in rows) for path in sorted({row["path"] for row in rows})},
        "input_tokens": sum(row["input"] for row in rows),
        "output_tokens": sum(row["output"] for row in rows),
        "prefill_ms_total": prefill_ms,
        "prefill_tokens_per_second": sum(row["input"] for row in rows) * 1000 / prefill_ms if prefill_ms else None,
        "prefill_ms_median": statistics.median(row["prefillMs"] for row in rows) if rows else None,
        "decode_intervals": decode_tokens,
        "decode_ms_total": decode_ms,
        "decode_tokens_per_second": decode_tokens * 1000 / decode_ms if decode_ms else None,
        "pi_wall_seconds": pi["wall_seconds"]["total"],
        "pi_completed_tasks": pi["completion_count"],
        "pi_task_count": pi["count"],
        "speed_definition": "sum worker input tokens / sum worker prefill seconds; sum worker decode intervals / sum worker decode seconds",
        "interpretation": "worker timings, not Pi application throughput; incomplete coverage is diagnostic only" if not complete else "worker timings, not Pi application throughput",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Pi HumanEval output directory")
    parser.add_argument("--serve-log", type=Path, action="append", required=True, help="Mac split-serve log; repeat for rotated logs")
    parser.add_argument("--mac-commit", required=True, help="Verified full commit SHA of running Mac worker")
    parser.add_argument("--windows-commit", help="Verified full commit SHA of running Windows agent, if used")
    parser.add_argument("--change", required=True, help="Brief description of measured change")
    parser.add_argument("--output", type=Path, required=True, help="Performance summary JSON")
    args = parser.parse_args()
    if not SHA.fullmatch(args.mac_commit) or args.windows_commit and not SHA.fullmatch(args.windows_commit):
        parser.error("Commit IDs must be verified full 40-character SHA-1 values")
    result = summarize(args.run, args.serve_log, args.mac_commit.lower(),
                       args.windows_commit.lower() if args.windows_commit else None, args.change)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["metrics_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
