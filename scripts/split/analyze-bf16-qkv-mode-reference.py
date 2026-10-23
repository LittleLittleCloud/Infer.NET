#!/usr/bin/env python3
"""Compare CUDA token/layer QKV in FP32 and BF16 modes against CPU products.

NumPy is an offline diagnostic dependency, not part of the Infer.NET runtime.
This checks one synthetic token's projection operands and accumulation; it is
not the independent full-model acceptance reference or a throughput test.
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib

import numpy as np


def helper():
    path = pathlib.Path(__file__).with_name("analyze-bf16-stage-reference.py")
    spec = importlib.util.spec_from_file_location("bf16_stage_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--cuda", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a fresh output file")
    ref = helper()
    profile = json.loads((args.model / "config.json").read_text())["quantization"]
    if (profile["bits"], profile["group_size"], profile["mode"]) != (8, 64, "affine"):
        raise ValueError("Expected affine8/group64 model")
    if ref.MODEL_REVISION not in (args.model / "provenance.json").read_text():
        raise ValueError("Unexpected model revision")
    result = json.loads((args.cuda / "result.json").read_text())
    if (result["tokens"], result["weightBits"], result["kv"],
            result["baseline"]["mode"], result["candidate"]["mode"],
            result["stageCompareModes"]) != (128, 8, "bf16", 5, 6, True):
        raise ValueError("Expected 128-token CUDA mode-5/mode-6 stage probe")
    token = result["stageProbeToken"]
    if not isinstance(token, int) or not 0 <= token < 32:
        raise ValueError("Stage probe token must be in first chunk")
    token_ids = [100 + (i * 137) % 247000 for i in range(128)]
    if result["tokensSha256"] != hashlib.sha256(json.dumps(token_ids).encode()).hexdigest():
        raise ValueError("Unexpected diagnostic token IDs")
    x5 = ref.stage(args.cuda, "cuda-mode5", "embedding-norm", 2048)
    x6 = ref.stage(args.cuda, "cuda-mode6", "embedding-norm", 2048)
    if x5.tobytes() != x6.tobytes():
        raise ValueError("QKV inputs differ before projection")
    actual5 = ref.stage(args.cuda, "cuda-mode5", "qkv", 8192)
    actual6 = ref.stage(args.cuda, "cuda-mode6", "qkv", 8192)
    shards = ref.Shards(args.model)
    packed = shards.tensor(ref.QKV + "weight").view(np.uint8).reshape(8192, 2048)
    scales = ref.bf16_to_f32(shards.tensor(ref.QKV + "scales"))
    biases = ref.bf16_to_f32(shards.tensor(ref.QKV + "biases"))
    fp32_reference = np.empty(8192, np.float64)
    bf16_reference = np.empty(8192, np.float64)
    rounded_input = ref.round_bf16(x5)
    for begin in range(0, 8192, 256):
        end = begin + 256
        weights = packed[begin:end].astype(np.float32) * np.repeat(scales[begin:end], 64, axis=-1)
        weights += np.repeat(biases[begin:end], 64, axis=-1)
        fp32_reference[begin:end] = weights.astype(np.float64) @ x5.astype(np.float64)
        bf16_reference[begin:end] = ref.round_bf16(weights).astype(np.float64) @ rounded_input.astype(np.float64)
    observed_shift = actual6.astype(np.float64) - actual5.astype(np.float64)
    predicted_shift = bf16_reference - fp32_reference
    summary = {
        "kind": "isolated-cuda-token-qkv-mode-cpu-product-reference",
        "token": token, "layer": 0,
        "inputBitIdenticalBeforeModeRounding": True,
        "inputValuesChangedByBf16Rounding": int(np.count_nonzero(x5 != rounded_input)),
        "fp32ModeVsOwnInputReference": ref.difference(actual5, fp32_reference),
        "bf16ModeVsOwnInputReference": ref.difference(actual6, bf16_reference),
        "bf16VsFp32CpuReference": ref.difference(bf16_reference, fp32_reference),
        "observedBf16VsFp32Mode": ref.difference(actual6, actual5),
        "unexplainedModeShift": ref.difference(observed_shift, predicted_shift),
        "limitations": ["One synthetic token and first layer", "CPU dot uses float64 accumulation on reconstructed operands",
                        "This does not establish independent full-model FP32 parity, logits, serving speed or memory safety"],
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
