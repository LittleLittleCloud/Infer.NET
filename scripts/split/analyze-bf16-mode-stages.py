#!/usr/bin/env python3
"""Locate the first CUDA arithmetic stage that differs between modes 5 and 6.

This is an isolated same-GPU diagnostic, not the independent FP32 reference.
Stage readbacks make elapsed time unsuitable for throughput comparisons.
"""
import argparse
import array
import hashlib
import json
import math
import pathlib
import sys


STAGES = {
    "embedding": 2048,
    "embedding-norm": 2048,
    "qkv": 8192,
    "delta-norm": 4096,
    "output-projection": 2048,
    "residual-norm": 2048,
    "router": 256,
    "shared-gate": 512,
    "expert-gate": 4096,
    "expert-down": 16384,
    "shared-down": 2048,
    "merge": 2048,
}


def read(path, count):
    raw = path.read_bytes()
    if len(raw) != count * 4:
        raise ValueError(f"Stage probe size mismatch: {path}")
    values = array.array("f")
    values.frombytes(raw)
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"Nonfinite stage value: {path}")
    return values, hashlib.sha256(raw).hexdigest()


def main():
    if sys.byteorder != "little":
        raise SystemExit("Stage probes require little-endian floats")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a fresh output file")
    result = json.loads((args.cuda / "result.json").read_text())
    if (result["tokens"], result["weightBits"],
            result["baseline"]["mode"], result["candidate"]["mode"],
            result.get("stageCompareModes")) != (128, 8, 5, 6, True) or result["kv"] not in ("bf16", "fp32"):
        raise ValueError("Expected same-input CUDA mode-5/mode-6 stage probe")
    token = result["stageProbeToken"]
    if not isinstance(token, int) or not 0 <= token < 32:
        raise ValueError("Invalid stage probe token")
    stages = []
    for name, count in STAGES.items():
        baseline, baseline_sha = read(args.cuda / f"cuda-mode5-{name}.f32", count)
        candidate, candidate_sha = read(args.cuda / f"cuda-mode6-{name}.f32", count)
        delta2 = sum((float(y) - float(x)) ** 2 for x, y in zip(baseline, candidate))
        ref2 = sum(float(x) ** 2 for x in baseline)
        stages.append({"stage": name, "floats": count,
                       "baselineSha256": baseline_sha, "candidateSha256": candidate_sha,
                       "relativeL2": math.sqrt(delta2 / max(ref2, 1e-30)),
                       "maxAbs": max(abs(float(y) - float(x)) for x, y in zip(baseline, candidate)),
                       "exact": baseline_sha == candidate_sha})
    summary = {"kind": "isolated-cuda-fp32-mode-versus-bf16-mode-stage-probe",
               "kv": result["kv"],
               "candidateQkvCompensation": result.get("candidateQkvCompensation", "none"),
               "token": token, "layer": 0, "stages": stages,
               "firstNonExactStage": next((item["stage"] for item in stages if not item["exact"]), None),
               "limitations": ["Synthetic token IDs", "CUDA mode 5 is an internal baseline, not the independent FP32 reference",
                               "Readbacks distort timing; no logits, transfer, suffix, decode or long-context acceptance"]}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"token": token, "firstNonExactStage": summary["firstNonExactStage"],
                      "stages": [{"stage": item["stage"], "relativeL2": item["relativeL2"]} for item in stages]}))


if __name__ == "__main__":
    main()
