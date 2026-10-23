#!/usr/bin/env python3
"""Check whether CUDA BF16 K/V exports are rounded FP32-KV exports."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def rounded_bf16_bits(values: np.ndarray) -> np.ndarray:
    words = values.view(np.uint32)
    return ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype('<u2')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp32-kv', type=Path, required=True)
    parser.add_argument('--bf16-kv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    fp32 = json.loads((args.fp32_kv / 'result.json').read_text())
    bf16 = json.loads((args.bf16_kv / 'result.json').read_text())
    if (fp32['kv'] != 'fp32' or bf16['kv'] != 'bf16'
            or fp32['candidateGemm'] != bf16['candidateGemm']
            or fp32['candidateProjectionCompensation'] != bf16['candidateProjectionCompensation']
            or fp32['candidateRouterProjection'] != bf16['candidateRouterProjection']
            or fp32['candidateSharedProjection'] != bf16['candidateSharedProjection']
            or any(fp32[key] != bf16[key] for key in
                   ('source', 'librarySha256', 'bundleSha256', 'tokensSha256', 'tokens'))):
        raise ValueError('Expected matched source, DLL, tokens and candidate profile, differing only in KV dtype')
    rows = []
    for layer in (3, 7, 11, 15, 19):
        for kind, label in ((0, 'key'), (1, 'value')):
            fp32_path = args.fp32_kv / f'candidate-state-{layer}-{kind}.bin'
            bf16_path = args.bf16_kv / f'candidate-state-{layer}-{kind}.bin'
            original = np.fromfile(fp32_path, dtype='<f4')
            stored = np.fromfile(bf16_path, dtype='<u2')
            if original.size != 128 * 512 or stored.size != original.size:
                raise ValueError(f'Unexpected state length at layer {layer} {label}')
            rounded = rounded_bf16_bits(original)
            expanded = (stored.astype(np.uint32) << 16).view(np.float32)
            difference = expanded.astype(np.float64) - original.astype(np.float64)
            rows.append({
                'layer': layer, 'state': label, 'values': int(stored.size),
                'differentFromRoundedFp32': int(np.count_nonzero(stored != rounded)),
                'relativeL2VsFp32': float(np.linalg.norm(difference) /
                                         max(np.linalg.norm(original.astype(np.float64)), 1e-30)),
                'maxAbsVsFp32': float(np.max(np.abs(difference))),
                'fp32Sha256': sha(fp32_path), 'bf16Sha256': sha(bf16_path),
            })
    output = {
        'kind': 'same-source-cuda-bf16-kv-rounding-diagnostic-not-e2e',
        'source': fp32['source'], 'cudaDllSha256': fp32['librarySha256'],
        'tokensSha256': fp32['tokensSha256'], 'rows': rows,
        'limitations': ['First attention layer isolates storage rounding; later states also include propagated activation error',
                        'No independent state oracle, suffix, logits, RPC or throughput'],
    }
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'firstAttentionExactRounding': all(row['differentFromRoundedFp32'] == 0
                                                      for row in rows if row['layer'] == 3),
                      'laterDifferentWords': sum(row['differentFromRoundedFp32']
                                                 for row in rows if row['layer'] > 3)}))


if __name__ == '__main__':
    main()
