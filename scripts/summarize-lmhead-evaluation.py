#!/usr/bin/env python3
"""Strict #103 summary. Missing clocks/coverage stay null; no adoption inference."""
import argparse
import json
import math
from pathlib import Path
import statistics

from issue103_common import (
    BATTERY_ASSISTED_WARNING, COHORT, DATASET, IMAGE, MAIN, METAL_LIMIT, PACKED_BYTES, PILOT, SHAPES, SWAP_LIMIT, TOOLS, TOOLS_SHA,
    aggregate, instant, metric_from_observation, read_json, reconcile, require, rows, save, sha,
)


def finite_duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def describe(values, complete=True):
    if not complete or not values or not all(finite_duration(v) for v in values):
        return {"count": len(values), "complete": False, "median": None, "mean": None, "min": None, "max": None}
    return {"count": len(values), "complete": True, "median": statistics.median(values), "mean": statistics.mean(values),
            "min": min(values), "max": max(values)}


def relative_change(value, baseline):
    if not finite_duration(value) or not finite_duration(baseline) or baseline <= 0:
        return None
    return 100 * (value / baseline - 1)


def fixed_summary(run):
    result, errors = [], []
    for arm in ("main", "on"):
        for n, m in SHAPES:
            path = run / f"fixed-{n}-{m}-{arm}"
            samples = []
            metrics = []
            valid = True
            for trial in range(1, 4):
                filename = path / f"trial-{trial}.json"
                observation = path / f"trial-{trial}-observation.json"
                if not filename.exists() or not observation.exists():
                    valid = False
                    continue
                row, observed = read_json(filename), read_json(observation)
                response = row.get("result") or {}
                swap = observed.get("newSwapOutBytes")
                okay = (not row.get("excludedWarmup") and not row.get("error") and
                        row.get("inputTokens") == n and row.get("outputLimit") == m and
                        response.get("promptTokens") == n and len(response.get("tokenIds") or []) == m and
                        row.get("deliveredTokenIds") == response.get("tokenIds") and
                        response.get("finishReason") == "length" and observed.get("outcome") == "completed" and
                        observed.get("inputTokens") == n and observed.get("outputTokens") == m and
                        type(swap) is int and 0 <= swap <= SWAP_LIMIT)
                timing = response.get("timings") or {}
                allocation = timing.get("metalAllocatedBytes")
                okay = okay and type(allocation) is int and 0 < allocation <= METAL_LIMIT
                valid &= okay
                samples.append(row)
                metrics.append(metric_from_observation(observed, MAIN if arm == "main" else "candidate"))
            valid &= len(samples) == 3 and len({tuple((s.get("result") or {}).get("tokenIds", [])) for s in samples}) == 1
            measured = aggregate(metrics, valid)
            if not measured["complete"]:
                errors.append(f"{arm} {n}/{m}: missing/invalid formal coverage")
            result.append({
                "arm": arm, "inputTokens": n, "outputLimit": m, "formalSamples": len(samples),
                "warmupsExcluded": True, "worker": measured,
                "clientWallSeconds": describe([s.get("clientWallSeconds") for s in samples], valid),
                "clientTtftMs": describe([s.get("clientTtftMs") for s in samples], valid),
                "swapOutBytes": [read_json(path / f"trial-{i}-observation.json")["newSwapOutBytes"]
                                 for i in range(1, 4) if (path / f"trial-{i}-observation.json").exists()],
                "workerDurations": {"prefillMs": describe([m["prefillMs"] for m in metrics]),
                                    "decodeMs": describe([m["decodeMs"] for m in metrics])} if measured["complete"] else None,
            })
    for n, m in SHAPES:
        left, right = run / f"fixed-{n}-{m}-main/trial-1.json", run / f"fixed-{n}-{m}-on/trial-1.json"
        if left.exists() and right.exists():
            if (read_json(left).get("result") or {}).get("tokenIds") != (read_json(right).get("result") or {}).get("tokenIds"):
                errors.append(f"{n}/{m}: main/on exact-sequence divergence")
                for entry in result:
                    if (entry["inputTokens"], entry["outputLimit"]) == (n, m):
                        entry["worker"] = aggregate([], False)
    return result, errors


def human_summary(run, cohort, arm, candidate):
    path = run / f"{cohort}-{arm}"
    if not (path / "pi/tasks.jsonl").exists():
        return {"cohort": cohort, "arm": arm, "available": False, "worker": aggregate([], False)}
    tasks = rows(path / "pi/tasks.jsonl")
    observations = rows(path / "observer/requests.jsonl")
    build = MAIN if arm == "main" else candidate
    expected_ids = ["HumanEval/" + str(i) for i in (PILOT if cohort == "pilot" else COHORT)]
    checked = reconcile(tasks, observations, build)
    if [t["task_id"] for t in tasks] != expected_ids:
        checked["problems"].append("Task denominator/order mismatch")
        checked["metrics"] = aggregate([], False)
    standard = read_json(path / "standard-performance.json") if (path / "standard-performance.json").exists() else None
    if standard is None or not standard.get("metrics_complete"):
        checked["problems"].append("Pinned standard summary unavailable/incomplete")
        checked["metrics"] = aggregate([], False)
    elif checked["metrics"]["complete"]:
        for key, standard_key in (("prefillTokensPerSecond", "prefill_tokens_per_second"),
                                  ("decodeTokensPerSecond", "decode_tokens_per_second")):
            if checked["metrics"][key] != standard.get(standard_key):
                checked["problems"].append("Independent/pinned summarizer disagreement")
                checked["metrics"] = aggregate([], False)
                break
    selected = [r for r in observations if any(instant(t["start_utc"]) <=
                instant(r.get("workerCompletedUtc", r["completedUtc"])) <= instant(t["end_utc"]) for t in tasks)]
    ttfts = [r["clientTtftMs"] for r in selected if r.get("outcome") == "completed" and r.get("clientTtftMs") is not None]
    official = read_json(path / "official.json") if (path / "official.json").exists() else None
    controls = read_json(path / "official-controls.json") if (path / "official-controls.json").exists() else {}
    negative, positive = controls.get("negative") or {}, controls.get("positive") or {}
    assertions_valid = bool(official and official.get("count") == len(expected_ids) and
        official.get("image") == IMAGE and official.get("dataset_sha256") == DATASET and
        [r.get("task_id") for r in official["rows"]] == expected_ids and
        negative.get("exit_code") == 1 and "AssertionError" in negative.get("stderr", "") and
        "/work/case.py" in negative.get("stderr", "") and positive.get("exit_code") == 0 and
        not any(r.get("exit_code") in (125, 126, 127) for r in official["rows"]))
    expected_requests = sum(len(t["pi"]["requests"]) for t in tasks)
    walls = [t.get("wall_seconds") for t in tasks]
    return {"cohort": cohort, "arm": arm, "available": True, "taskDenominator": len(expected_ids),
            "tasksObserved": len(tasks), "completedTasks": sum(bool(t.get("completed")) for t in tasks),
            "timeouts": sum(bool(t.get("timeout")) for t in tasks),
            "taskFailures": [t["task_id"] for t in tasks if not t.get("completed")],
            "expectedRequests": expected_requests, "observedRequests": len(selected),
            "worker": checked["metrics"], "coverageProblems": checked["problems"],
            "ttftBoundary": "Outer observer forwarding-to-first-valid-text/tool delta; Pi-to-observer dispatch excluded",
            "clientTtftMs": describe(ttfts, len(ttfts) == expected_requests),
            "ttftCoverage": {"obtained": len(ttfts), "expected": expected_requests},
            "piWallSeconds": sum(walls) if all(finite_duration(v) for v in walls) else None,
            "officialPassed": official["passed"] if assertions_valid else None,
            "officialPassRate": official["passed"] / len(expected_ids) if assertions_valid else None,
            "officialInfrastructureValid": assertions_valid,
            "standardSummarySha256": sha(path / "standard-performance.json") if standard else None}


def fixed_comparisons(fixed):
    results = []
    for n, m in SHAPES:
        entries = {row["arm"]: row for row in fixed if (row["inputTokens"], row["outputLimit"]) == (n, m)}
        main, candidate = entries["main"], entries["on"]
        row = {"inputTokens": n, "outputLimit": m, "prefillLatencyChangePercent": None,
               "decodeThroughputChangePercent": None, "prefillRegressionWithin5Percent": None,
               "decodeRegressionWithin5Percent": None}
        if main["worker"]["complete"] and candidate["worker"]["complete"]:
            left, right = main["workerDurations"], candidate["workerDurations"]
            prefill_ratio = right["prefillMs"]["median"] / left["prefillMs"]["median"]
            decode_ratio = left["decodeMs"]["median"] / right["decodeMs"]["median"]
            row.update(prefillLatencyChangePercent=100 * (prefill_ratio - 1),
                       decodeThroughputChangePercent=100 * (decode_ratio - 1),
                       prefillRegressionWithin5Percent=prefill_ratio <= 1.05,
                       decodeRegressionWithin5Percent=decode_ratio >= .95)
        results.append(row)
    return results


def summarize(run):
    plan = read_json(run / "plan.json")
    fixed, fixed_errors = fixed_summary(run)
    humans = [human_summary(run, cohort, arm, plan["candidate"])
              for cohort in ("pilot", "cohort") for arm in ("main", "off", "on")]
    comparisons = {}
    formal = {r["arm"]: r for r in humans if r["cohort"] == "cohort"}
    for arm in ("off", "on"):
        main, candidate = formal["main"]["worker"], formal[arm]["worker"]
        if main["complete"] and candidate["complete"]:
            comparisons[arm] = {
                "decodeChangePercent": relative_change(candidate["decodeTokensPerSecond"], main["decodeTokensPerSecond"]),
                "prefillChangePercent": relative_change(candidate["prefillTokensPerSecond"], main["prefillTokensPerSecond"])}
        else:
            comparisons[arm] = {"decodeChangePercent": None, "prefillChangePercent": None}
    restoration = read_json(run / "restoration.json") if (run / "restoration.json").exists() else None
    on, off = comparisons["on"], comparisons["off"]
    main_passed, on_passed = formal["main"].get("officialPassed"), formal["on"].get("officialPassed")
    return {"schema": "issue103-evaluation-summary-v1", "main": MAIN, "candidate": plan["candidate"],
            "powerPolicy": plan.get("powerPolicy"), "powerThresholds": plan.get("powerThresholds"),
            "powerWarnings": [BATTERY_ASSISTED_WARNING] if plan.get("powerPolicy") == "battery-assisted-35w" else [],
            "toolsCommit": TOOLS_SHA, "toolHashes": TOOLS, "fixed": fixed, "fixedProblems": fixed_errors,
            "fixedMedianVersusMain": fixed_comparisons(fixed),
            "human": humans, "formal20VersusMain": comparisons,
            "predeclaredChecks": {
                "formal20DecodeGainAtLeast5Percent": on["decodeChangePercent"] >= 5 if on["decodeChangePercent"] is not None else None,
                "candidateOffPrefillWithin5Percent": off["prefillChangePercent"] >= -5 if off["prefillChangePercent"] is not None else None,
                "candidateOffDecodeWithin5Percent": off["decodeChangePercent"] >= -5 if off["decodeChangePercent"] is not None else None,
                "officialOnPassesNotBelowMain": on_passed >= main_passed if on_passed is not None and main_passed is not None else None,
            },
            "restoration": restoration, "candidatePackedPayloadBytes": PACKED_BYTES,
            "adoptionDecision": "Requires parent review of all correctness/resource/quality/coverage evidence; no automatic adoption",
            "limitations": ["Full checkpoint logits/internal states unavailable from the streaming API.",
                            "No physical GPU residency high-water query; request allocation is device allocation, not residency.",
                            "Missing/error durations or token coverage invalidate aggregate worker metrics, not filled with zeros.",
                            "Pi wall, observer TTFT and worker N-1 decode clocks are distinct."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local-only JSON output")
    args = parser.parse_args()
    run = args.run.resolve()
    require(run in args.output.resolve().parents, "Summary must remain in the private run directory")
    result = summarize(run)
    save(args.output, result)
    print(json.dumps({"written": True, "fixedProblems": len(result["fixedProblems"]),
                      "formal20VersusMain": result["formal20VersusMain"],
                      "powerPolicy": result["powerPolicy"], "powerWarnings": result["powerWarnings"]}, allow_nan=False))


if __name__ == "__main__":
    main()
