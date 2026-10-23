#!/usr/bin/env python3
"""Compare isolated token-0/layer-0 BF16 stage probes from Metal and CUDA.

This locates the first differing arithmetic boundary. It is not an inference
quality or throughput acceptance test; probes force readbacks and submissions.
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


def read(path: pathlib.Path, count: int):
    raw = path.read_bytes()
    if len(raw) != count * 4:
        raise ValueError(f"{path}: expected {count * 4} bytes, got {len(raw)}")
    values = array.array("f")
    values.frombytes(raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: non-finite stage value")
    return values, hashlib.sha256(raw).hexdigest()


def main():
    if sys.byteorder != "little":
        raise SystemExit("Stage probes require little-endian floats")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metal", required=True, type=pathlib.Path)
    parser.add_argument("--cuda", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a fresh output file")
    metal_result = json.loads((args.metal / "result.json").read_text())
    cuda_result = json.loads((args.cuda / "result.json").read_text())
    if (metal_result["tokens"] != cuda_result["tokens"] or
            metal_result["tokens"] != 128 or
            metal_result["kv"] != cuda_result["kv"] or
            metal_result["kv"] != "bf16" or
            metal_result["weightBits"] != cuda_result["weightBits"] or
            metal_result["weightBits"] != 8):
        raise ValueError("Probe model/token/precision profiles do not match")
    token_ids = [100 + (i * 137) % 247000 for i in range(128)]
    if (json.loads((args.metal / "tokens.json").read_text()) != token_ids or
            cuda_result["tokensSha256"] != hashlib.sha256(json.dumps(token_ids).encode()).hexdigest() or
            metal_result["routerProjection"] != cuda_result["candidateRouterProjection"] or
            cuda_result["candidateSharedProjection"] != "fp32-diagnostic"):
        raise ValueError("Probe token IDs or router/shared projection profiles differ")
    stages = []
    for name, count in STAGES.items():
        metal, metal_sha = read(args.metal / f"metal-{name}.f32", count)
        cuda, cuda_sha = read(args.cuda / f"cuda-{name}.f32", count)
        diff2 = sum((float(x) - float(y)) ** 2 for x, y in zip(metal, cuda))
        ref2 = sum(float(x) ** 2 for x in metal)
        stages.append({"stage": name, "floats": count,
                       "metalSha256": metal_sha, "cudaSha256": cuda_sha,
                       "relativeL2": math.sqrt(diff2 / max(ref2, 1e-30)),
                       "maxAbs": max(abs(float(x) - float(y)) for x, y in zip(metal, cuda)),
                       "exact": metal_sha == cuda_sha})
    result = {"kind": "isolated-token-zero-layer-zero-bf16-arithmetic-probe",
              "tokens": 128, "weightBits": 8, "kv": "bf16", "stages": stages,
              "limitations": ["Synthetic token IDs", "Readbacks disturb timing",
                              "No transfer, suffix, decode, logits, or long-context acceptance"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"firstNonExact": next((s for s in stages if not s["exact"]), None),
                      "stages": [{"stage": s["stage"], "relativeL2": s["relativeL2"],
                                  "maxAbs": s["maxAbs"]} for s in stages]}))


if __name__ == "__main__":
    main()
