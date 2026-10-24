#!/usr/bin/env python3
"""Import the separately observed Issue #54 HumanEval retake without rewriting its old run."""

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "issue54-bf16-2048-ttft-human20-20261024"
CONFIG_ID = "pi-humaneval20-issue54-bf16-2048-observed-wifi"
REPORT = "docs/EXPERIMENT-54-TTFT-RETAKE-20261024.md"
RESULTS_PR = "https://github.com/LittleLittleCloud/Infer.NET/pull/57"
SOURCE = "https://github.com/LittleLittleCloud/Infer.NET/blob/main/site/trends.json"
OLD_ID = "issue54-bf16-2048-human20-20261024"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def import_run(manifest, run_dir, harness_commit):
    old = next(run for run in manifest["runs"] if run["id"] == OLD_ID)
    assert not any(run["id"] == RUN_ID for run in manifest["runs"])
    assert not any(config["id"] == CONFIG_ID for config in manifest["configurations"])
    old_config = next(config for config in manifest["configurations"]
                      if config["id"] == old["configurationId"])
    cohort = run_dir / "cohort-20"
    observed_path = run_dir / "cohort-observer/requests.jsonl"
    cohort_manifest = read_json(cohort / "manifest.json")
    summary = read_json(cohort / "summary.json")
    task_rows = [json.loads(line) for line in (cohort / "tasks.jsonl").read_text().splitlines() if line]
    observations = [json.loads(line) for line in observed_path.read_text().splitlines() if line]
    assert len(task_rows) == 20 and len(observations) == summary["request_count"]
    assert cohort_manifest["ids"] == old_config["plannedTasks"]
    assert cohort_manifest["dataset_sha256"] == old_config["datasetSha256"]
    assert cohort_manifest["pi_version"] == "1.4.0" and cohort_manifest["seed"] == 0
    assert cohort_manifest["context_window"] == 32768 and cohort_manifest["max_tokens"] == 4096
    assert cohort_manifest["model_revision"] == old_config["modelRevision"]
    assert cohort_manifest["endpoint_build"] == old["measuredCommit"]
    assert summary["tokens"]["cacheRead"] == summary["tokens"]["cacheWrite"] == 0
    assert all(task["completed"] for task in task_rows), "Keep incomplete cohort as local evidence"
    assert all(row["outcome"] == "completed" and row["firstTokenReceived"] is True
               and row["clientTtftMs"] is not None and row["workerResult"] is not None
               and row["workerResult"]["executionPath"] == "cuda-metal-split"
               for row in observations), "Every request needs real split worker and first-token timing"
    assert all(row["prefillSeconds"] is not None and
               (row["outputTokens"] < 2 or row["decodeSeconds"] is not None)
               for row in observations), "Every eligible worker interval needs timing"
    assert sum(row["inputTokens"] for row in observations) == summary["tokens"]["input"]
    assert sum(row["outputTokens"] for row in observations) == summary["tokens"]["output"]

    config = copy.deepcopy(old_config)
    config.update({"id": CONFIG_ID,
                   "label": "Issue #54 · BF16/2048 · Pi fixed 20 · observed client TTFT · Wi-Fi",
                   "workload": old_config["workload"] +
                   "; two loopback streaming observation hops, separate retake",
                   "timing": {**old_config["timing"],
                              "prefill": "raw worker ServiceEvent.result prefill timings; observation hops disclosed separately"},
                   "latestBaselineRun": None})
    run_index = len(manifest["runs"])
    tasks, requests = [], []
    for task_index, task in enumerate(task_rows):
        matched = [row for row in observations if timestamp(task["start_utc"])
                   <= timestamp(row["completedUtc"]) <= timestamp(task["end_utc"])]
        assert len(matched) == len(task["pi"]["requests"]), task["task_id"]
        assert sum(row["inputTokens"] for row in matched) == sum(
            row["usage"]["input"] for row in task["pi"]["requests"])
        assert sum(row["outputTokens"] for row in matched) == sum(
            row["usage"]["output"] for row in task["pi"]["requests"])
        tasks.append({"taskId": task["task_id"], "attempt": 1,
                      "wallSeconds": task["wall_seconds"],
                      "inputTokens": sum(row["inputTokens"] for row in matched),
                      "outputTokens": sum(row["outputTokens"] for row in matched),
                      "modelRequests": len(matched), "status": "unverified",
                      "officialValidation": "unverified",
                      "source": f"{SOURCE}#/runs/{run_index}/tasks/{task_index}"})
        for row in matched:
            index = len(requests)
            requests.append({"id": f"issue54-bf16-2048-ttft-{index + 1:04d}",
                             "inputTokens": row["inputTokens"],
                             "outputTokens": row["outputTokens"],
                             "prefillSeconds": row["prefillSeconds"],
                             "decodeSeconds": row["decodeSeconds"],
                             "clientTtftMs": row["clientTtftMs"],
                             "ttftBoundary": "client-send-to-first-valid-output-token",
                             "firstTokenReceived": True, "outcome": "completed",
                             "taskId": task["task_id"], "attempt": 1,
                             "source": f"{SOURCE}#/runs/{run_index}/requests/{index}"})
    assert len(requests) == len(observations)
    prefill = sum(row["inputTokens"] for row in requests) / sum(row["prefillSeconds"] for row in requests)
    decodable = [row for row in requests if row["outputTokens"] >= 2]
    decode = sum(row["outputTokens"] - 1 for row in decodable) / sum(
        row["decodeSeconds"] for row in decodable)
    ttft = statistics.median(row["clientTtftMs"] for row in requests)
    evidence_files = [cohort / name for name in ("manifest.json", "tasks.jsonl", "summary.json")]
    evidence_files += [observed_path, run_dir / "preflight.json", run_dir / "observer-script.sha256",
                       run_dir / "cohort-20.log", run_dir / "candidate-serve.log"]
    evidence = {str(Path("artifacts/dual-host") / run_dir.name / path.relative_to(run_dir)):
                {"sha256": sha256(path), "availability": "Local ignored artifacts; not a public download"}
                for path in evidence_files}
    run = copy.deepcopy(old)
    run.update({
        "id": RUN_ID, "configurationId": CONFIG_ID,
        "title": "Issue #54 · BF16/2048 · Pi fixed 20 · TTFT retake",
        "date": datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d America/Los_Angeles"),
        "state": "not-adopted",
        "change": "Same BF16 dense GEMM / CUDA outer chunk 2048 candidate; separate Pi retake with two loopback observers measuring first valid text/tool output at the proxy client boundary.",
        "result": (f"20/20 Pi tasks completed; official assertions not run. "
                   f"{len(requests)}/{len(requests)} client TTFT samples, median {ttft:.2f} ms. "
                   f"Worker prefill {prefill:.2f} tok/s and decode {decode:.2f} tok/s; "
                   f"Pi wall {summary['wall_seconds']['total']:.2f} s. "
                   "This is a new observed run; original missing-TTFT run remains unchanged."),
        "conditions": {**old["conditions"], "observer": "Two loopback streaming observation hops; client boundary is observer proxy send to first valid text/tool delta; Pi-to-proxy dispatch excluded",
                       "power": "Mac AC", "warmup": "One separate short split request before the 5-task pilot and 20-task cohort"},
        "binaryEvidence": f"https://github.com/LittleLittleCloud/Infer.NET/blob/main/{REPORT}",
        "harnessCommit": harness_commit, "harnessSha256": cohort_manifest["harness_sha256"],
        "requestCount": len(requests), "requests": requests, "tasks": tasks,
        "reportedTotals": {"wallSeconds": summary["wall_seconds"]["total"],
                           "inputTokens": summary["tokens"]["input"],
                           "outputTokens": summary["tokens"]["output"],
                           "modelRequests": len(requests)},
        "rawEvidence": evidence,
        "links": [link for link in old["links"] if link["role"] not in ("results-pr", "report")] +
                 [{"role": "results-pr", "label": "TTFT retake results PR #57",
                   "url": RESULTS_PR},
                  {"role": "report", "label": "TTFT retake and raw evidence hashes",
                   "url": f"https://github.com/LittleLittleCloud/Infer.NET/blob/main/{REPORT}"}],
    })
    manifest["configurations"].append(config)
    manifest["runs"].append(run)
    return prefill, decode, ttft, len(requests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--harness-commit", required=True)
    args = parser.parse_args()
    path = ROOT / "site/trends.json"
    raw = read_json(path)
    values = import_run(raw, args.run_dir.resolve(), args.harness_commit)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
    print("Imported retake: prefill %.2f, decode %.2f tok/s, TTFT %.2f ms, %d requests" % values)


if __name__ == "__main__":
    main()
