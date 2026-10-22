#!/usr/bin/env python3
"""Compare independent FP32 oracle artifacts with our .NET/Metal outputs. Stdlib only."""
import argparse
import array
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/reference/qwen36')
args = parser.parse_args()
root = args.directory
reference = json.loads((root / 'reference.json').read_text())

def compare(name):
    def read(path):
        values = array.array('f')
        values.frombytes(path.read_bytes())
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f'Non-finite values: {path}')
        return values
    a, b = read(root / name), read(root / 'own' / name)
    if len(a) != len(b):
        raise ValueError(f'Shape mismatch: {name}')
    difference = [x-y for x, y in zip(a, b)]
    relative = math.sqrt(sum(x*x for x in difference) / max(sum(x*x for x in a), 1e-30))
    maximum = max(abs(x) for x in difference)
    if relative > 1e-4 or maximum > .01:
        raise ValueError(f'Numerical mismatch: {name}: relative L2 {relative}, max absolute {maximum}')
    return {'name': name, 'relativeL2': relative, 'maxAbsolute': maximum,
            'referenceArgmax': max(range(len(a)), key=a.__getitem__),
            'ownArgmax': max(range(len(b)), key=b.__getitem__)}

traces = [compare(f'p{p}-layer{layer}.f32') for p in reference['tracePositions'] for layer in range(40)]
logits = compare('logits.f32')
if logits['referenceArgmax'] != logits['ownArgmax']:
    raise ValueError('Final greedy argmax mismatch')
result = {'passed': True, 'toleranceRelativeL2': 1e-4, 'toleranceMaxAbsolute': .01,
          'traceCount': len(traces), 'worstTraceRelativeL2': max(t['relativeL2'] for t in traces),
          'worstTraceMaxAbsolute': max(t['maxAbsolute'] for t in traces), 'logits': logits, 'traces': traces}
(root / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='traces'}, indent=2))
