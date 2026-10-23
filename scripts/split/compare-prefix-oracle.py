#!/usr/bin/env python3
"""Compare fixed-token Metal/CUDA prefix traces with an independent FP32 oracle."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    value = np.fromfile(path, dtype='<f4')
    if value.size != int(np.prod(shape)):
        raise ValueError(f'{path}: expected {shape}, got {value.size} float32 values')
    return value.reshape(shape)


def check(name: str, expected: np.ndarray, actual: np.ndarray) -> dict:
    finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all())
    if not finite:
        return {'name': name, 'finite': False, 'pass': False}
    a = expected.astype(np.float64)
    b = actual.astype(np.float64)
    error = b - a
    relative = float(np.linalg.norm(error) / max(np.linalg.norm(a), 1e-30))
    absolute = float(np.max(np.abs(error)))
    return {'name': name, 'finite': True, 'relativeL2': relative,
            'maxAbs': absolute, 'pass': relative < 1e-4 and absolute < .01}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('oracle', 'metal', 'cuda', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    oracle = json.loads((args.oracle / 'result.json').read_text())
    metal = json.loads((args.metal / 'result.json').read_text())
    cuda = json.loads((args.cuda / 'result.json').read_text())
    if oracle['batchTokens'] != 1:
        raise ValueError('Only the validated serial MLX oracle is accepted')
    if (oracle['kind'] != 'independent-fp32-prefix-oracle-not-product-inference-or-benchmark'
            or oracle['dtype'] != metal['kv'] or metal['gemm'] != 'fp32'
            or cuda['kv'] not in ('fp32', 'bf16') or cuda['baselineGemm'] != 'fp32'
            or cuda['candidateGemm'] != 'bf16'
            or not (oracle['weightBits'] == metal['weightBits'] == cuda['weightBits'] == 8)):
        raise ValueError('Precision or weight profile mismatch')
    if oracle['tokensSha256'] != cuda['tokensSha256']:
        raise ValueError('Token ID hash mismatch')
    ids = json.loads((args.metal / 'tokens.json').read_text())
    if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != oracle['tokensSha256']:
        raise ValueError('Metal token IDs differ from oracle')
    shape = (128, 20, 2048)
    control_name = 'cudaFp32' if cuda['kv'] == 'fp32' else 'cudaFp32GemmBf16Kv'
    candidate_name = 'cudaBf16GemmFp32Kv' if cuda['kv'] == 'fp32' else 'cudaBf16GemmBf16Kv'
    traces = {
        'metalFp32': read(args.metal / 'trace.f32', shape),
        control_name: read(args.cuda / 'baseline-trace.f32', shape),
        candidate_name: read(args.cuda / 'candidate-trace.f32', shape),
    }
    results = {name: [] for name in traces}
    for layer in range(20):
        expected = read(args.oracle / f'layer-{layer}.f32', (128, 2048))
        if hashlib.sha256(expected.tobytes()).hexdigest() != oracle['layerSha256'][layer]:
            raise ValueError(f'Oracle layer {layer} hash mismatch')
        for name, trace in traces.items():
            results[name].append(check(f'layer-{layer}', expected, trace[:, layer, :]))
    summary = {name: {
        'passingLayers': sum(item['pass'] for item in checks),
        'totalLayers': len(checks),
        'firstFailure': next((item for item in checks if not item['pass']), None),
        'worstRelative': max(checks, key=lambda item: item.get('relativeL2', float('inf'))),
        'checks': checks,
    } for name, checks in results.items()}
    output = {
        'kind': 'offline-independent-fp32-prefix-layer-comparison-not-e2e',
        'tokens': 128,
        'modelRevision': oracle['modelRevision'],
        'cudaKv': cuda['kv'],
        'tokensSha256': oracle['tokensSha256'],
        'oracleLogitsSha256': oracle['logitsSha256'],
        'metalTraceSha256': metal['traceSha256'],
        'cudaBaselineTraceSha256': cuda['baseline']['traceSha256'],
        'cudaCandidateTraceSha256': cuda['candidate']['traceSha256'],
        'results': summary,
        'limitations': ['Layer outputs only; no oracle KV/Delta states',
                        'No CUDA suffix, logits, greedy tokens, real RPC, or 128K',
                        'Probe readbacks and separate runs invalidate throughput interpretation'],
    }
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({name: {key: value for key, value in row.items() if key != 'checks'}
                      for name, row in summary.items()}))


if __name__ == '__main__':
    main()
