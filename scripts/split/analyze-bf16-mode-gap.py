#!/usr/bin/env python3
"""Measure token-level layer-0 drift between CUDA FP32 mode 5 and BF16 mode 6.

This uses isolated synthetic prefix traces, not a formal independent reference
or a serving benchmark. It does not identify which operation caused the drift.
"""
import argparse
import array
import hashlib
import json
import math
import pathlib
import statistics
import sys


def load(path, expected_sha):
    raw = path.read_bytes()
    if len(raw) != 128 * 20 * 2048 * 4 or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"Trace size or SHA mismatch: {path}")
    values = array.array("f")
    values.frombytes(raw)
    return values


def main():
    if sys.byteorder != "little":
        raise SystemExit("Trace files require little-endian floats")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a fresh output file")
    result = json.loads((args.cuda / "result.json").read_text())
    if (result["tokens"], result["weightBits"], result["kv"],
            result["baseline"]["mode"], result["candidate"]["mode"]) != (128, 8, "bf16", 5, 6):
        raise ValueError("Expected 128-token mode-5/mode-6 diagnostic")
    baseline = load(args.cuda / "baseline-trace.f32", result["baseline"]["traceSha256"])
    candidate = load(args.cuda / "candidate-trace.f32", result["candidate"]["traceSha256"])
    per_token = []
    all_delta2 = all_ref2 = 0.0
    for token in range(128):
        offset = token * 20 * 2048
        delta2 = ref2 = 0.0
        for i in range(offset, offset + 2048):
            x, y = float(baseline[i]), float(candidate[i])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("Nonfinite layer-0 value")
            delta2 += (y - x) ** 2
            ref2 += x * x
        per_token.append(math.sqrt(delta2 / max(ref2, 1e-30)))
        all_delta2 += delta2
        all_ref2 += ref2
    summary = {
        "kind": "isolated-cuda-bf16-versus-fp32-mode-gap",
        "tokens": 128,
        "layer": 0,
        "baselineTraceSha256": result["baseline"]["traceSha256"],
        "candidateTraceSha256": result["candidate"]["traceSha256"],
        "aggregateRelativeL2": math.sqrt(all_delta2 / max(all_ref2, 1e-30)),
        "tokenRelativeL2Min": min(per_token),
        "tokenRelativeL2Median": statistics.median(per_token),
        "tokenRelativeL2Max": max(per_token),
        "worstToken": per_token.index(max(per_token)),
        "tokensAbove1e-4": sum(value >= 1e-4 for value in per_token),
        "tokensAbove1e-3": sum(value >= 1e-3 for value in per_token),
        "limitations": ["Synthetic token IDs", "CUDA mode 5 is an internal baseline, not an independent FP32 reference",
                        "Does not identify route changes or validate full model/output quality"]}
    expected = result["layerDrift"][0]["relativeL2"]
    if abs(summary["aggregateRelativeL2"] - expected) > 1e-10:
        raise ValueError("Trace calculation does not match saved layer drift")
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
