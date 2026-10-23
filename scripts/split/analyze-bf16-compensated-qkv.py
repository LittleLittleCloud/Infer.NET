#!/usr/bin/env python3
"""Offline BF16 high/low QKV product feasibility check for CUDA token 16.

This estimates the numerical value of extra BF16 products. It is neither a
CUDA kernel nor a throughput benchmark, and it does not validate a full layer.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def helper():
    path = Path(__file__).with_name('analyze-bf16-stage-reference.py')
    spec = importlib.util.spec_from_file_location('bf16_stage_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--cuda', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    ref = helper()
    info = json.loads((args.cuda / 'result.json').read_text())
    if (info['tokens'] != 128 or info['weightBits'] != 8 or info['stageProbeToken'] != 16
            or info['baseline']['mode'] != 5 or info['candidate']['mode'] != 6
            or not info['stageCompareModes']):
        raise ValueError('Expected the fixed token-16 FP32/BF16 CUDA stage probe')
    ids = [100 + (i * 137) % 247000 for i in range(128)]
    if info['tokensSha256'] != hashlib.sha256(json.dumps(ids).encode()).hexdigest():
        raise ValueError('Diagnostic token IDs differ')
    profile = json.loads((args.model / 'config.json').read_text())['quantization']
    if (profile['bits'], profile['group_size'], profile['mode']) != (8, 64, 'affine'):
        raise ValueError('Expected affine8/group64 weights')
    source = json.loads((args.model / 'provenance.json').read_text())
    if source['revision'] != ref.MODEL_REVISION:
        raise ValueError('Model revision differs')
    x = ref.stage(args.cuda, 'cuda-mode5', 'embedding-norm', 2048)
    observed_fp32 = ref.stage(args.cuda, 'cuda-mode5', 'qkv', 8192)
    observed_bf16 = ref.stage(args.cuda, 'cuda-mode6', 'qkv', 8192)
    x_hi = ref.round_bf16(x)
    x_lo = ref.round_bf16(x - x_hi)
    if not np.isfinite(x_lo).all():
        raise ValueError('Nonfinite low BF16 input plane')
    shards = ref.Shards(args.model)
    packed = shards.tensor(ref.QKV + 'weight').view(np.uint8).reshape(8192, 2048)
    scales = ref.bf16_to_f32(shards.tensor(ref.QKV + 'scales'))
    biases = ref.bf16_to_f32(shards.tensor(ref.QKV + 'biases'))
    reference = np.empty(8192, dtype=np.float64)
    one = np.empty_like(reference)
    two = np.empty_like(reference)
    three = np.empty_like(reference)
    four = np.empty_like(reference)
    for begin in range(0, 8192, 256):
        end = begin + 256
        weights = packed[begin:end].astype(np.float32) * np.repeat(scales[begin:end], 64, axis=-1)
        weights += np.repeat(biases[begin:end], 64, axis=-1)
        w_hi = ref.round_bf16(weights)
        w_lo = ref.round_bf16(weights - w_hi)
        reference[begin:end] = weights.astype(np.float64) @ x.astype(np.float64)
        main_product = w_hi.astype(np.float64) @ x_hi.astype(np.float64)
        w_correction = w_lo.astype(np.float64) @ x_hi.astype(np.float64)
        x_correction = w_hi.astype(np.float64) @ x_lo.astype(np.float64)
        cross = w_lo.astype(np.float64) @ x_lo.astype(np.float64)
        one[begin:end] = main_product
        two[begin:end] = main_product + x_correction
        three[begin:end] = main_product + x_correction + w_correction
        four[begin:end] = main_product + x_correction + w_correction + cross
    output = {
        'kind': 'offline-bf16-high-low-qkv-feasibility-not-gpu-performance',
        'modelRevision': ref.MODEL_REVISION,
        'token': 16,
        'layer': 0,
        'projection': 'qkv',
        'operandFormat': 'BF16 high/low planes, float64 diagnostic accumulation',
        'inputResidualMaxAbs': float(np.max(np.abs(x - x_hi - x_lo))),
        'gpuFp32VsReference': ref.difference(observed_fp32, reference),
        'gpuBf16VsReference': ref.difference(observed_bf16, reference),
        'oneBf16ProductVsReference': ref.difference(one, reference),
        'twoProductsInputCorrectedVsReference': ref.difference(two, reference),
        'threeProductsInputAndWeightCorrectedVsReference': ref.difference(three, reference),
        'fourProductsIncludingLowLowVsReference': ref.difference(four, reference),
        'limitations': ['One token and one QKV projection only',
                        'Float64 CPU reductions omit CUDA WMMA accumulation error',
                        'Extra products may increase GPU runtime and memory pressure',
                        'No complete layer, route, KV, logits, 128K or serving validation'],
    }
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key: value for key, value in output.items() if key.endswith('VsReference')}))


if __name__ == '__main__':
    main()
