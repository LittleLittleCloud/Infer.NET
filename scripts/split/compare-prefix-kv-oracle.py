#!/usr/bin/env python3
"""Compare CUDA BF16-KV prefix layers with an independent BF16-storage diagnostic.

The BF16-storage oracle deliberately differs from the fixed FP32 acceptance
reference. Passing this comparison can localize a KV error, not pass the gate.
"""

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
    a, b = expected.astype(np.float64), actual.astype(np.float64)
    relative = float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30))
    absolute = float(np.max(np.abs(b - a)))
    return {'name': name, 'finite': True, 'relativeL2': relative,
            'maxAbs': absolute, 'pass': relative < 1e-4 and absolute < .01}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('oracle', 'cuda', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    oracle = json.loads((args.oracle / 'result.json').read_text())
    cuda = json.loads((args.cuda / 'result.json').read_text())
    if (oracle['kind'] != 'independent-bf16-kv-fp32-attention-prefix-diagnostic-not-product-inference'
            or oracle['kvStorage'] != 'bf16' or oracle['dtype'] != 'fp32'
            or oracle['batchTokens'] != 1 or cuda['kv'] != 'bf16'
            or cuda['baselineGemm'] != 'fp32' or cuda['candidateGemm'] != 'bf16'
            or oracle['weightBits'] != cuda['weightBits'] or cuda['weightBits'] != 8
            or oracle['tokens'] != cuda['tokens'] or oracle['tokens'] != 128
            or oracle['tokensSha256'] != cuda['tokensSha256']):
        raise ValueError('Oracle, KV precision, weight or token profile mismatch')
    traces = {
        'cudaFp32GemmBf16Kv': read(args.cuda / 'baseline-trace.f32', (128, 20, 2048)),
        'cudaBf16GemmBf16Kv': read(args.cuda / 'candidate-trace.f32', (128, 20, 2048)),
    }
    checks = {name: [] for name in traces}
    for layer in range(20):
        expected = read(args.oracle / f'layer-{layer}.f32', (128, 2048))
        if hashlib.sha256(expected.tobytes()).hexdigest() != oracle['layerSha256'][layer]:
            raise ValueError(f'Oracle layer {layer} hash mismatch')
        for name, trace in traces.items():
            checks[name].append(check(f'layer-{layer}', expected, trace[:, layer, :]))
    results = {name: {
        'passingLayers': sum(item['pass'] for item in rows),
        'totalLayers': len(rows),
        'firstFailure': next((item for item in rows if not item['pass']), None),
        'worstRelative': max(rows, key=lambda item: item.get('relativeL2', float('inf'))),
        'checks': rows,
    } for name, rows in checks.items()}
    output = {
        'kind': 'offline-independent-bf16-kv-storage-comparison-not-fp32-acceptance-or-e2e',
        'modelRevision': oracle['modelRevision'],
        'tokensSha256': oracle['tokensSha256'],
        'oracleLogitsSha256': oracle['logitsSha256'],
        'cudaBaselineTraceSha256': cuda['baseline']['traceSha256'],
        'cudaCandidateTraceSha256': cuda['candidate']['traceSha256'],
        'results': results,
        'limitations': ['BF16 KV storage oracle is not the fixed FP32 reference',
                        'Layer outputs only; no independent K/V, Delta, suffix or logits comparison',
                        'No real RPC, 128K, memory or throughput acceptance'],
    }
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({name: {key: value for key, value in row.items() if key != 'checks'}
                      for name, row in results.items()}))


if __name__ == '__main__':
    main()
