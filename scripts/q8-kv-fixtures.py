#!/usr/bin/env python3
"""Freeze independent IEEE-half/block32 Q8 fixtures before native evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def encode(values):
    result = bytearray()
    for at in range(0, len(values), 32):
        block = values[at:at+32]
        if len(block) != 32 or not all(math.isfinite(v) for v in block):
            raise ValueError('Invalid block')
        maximum = max(abs(v) for v in block)
        half = struct.pack('<e', f32(maximum / 127.0))
        scale = struct.unpack('<e', half)[0]
        if not math.isfinite(scale) or (maximum and not scale):
            raise ValueError('Invalid scale')
        result += half
        result += struct.pack('32b', *(0 if not scale else max(-127, min(127, round(f32(v / scale)))) for v in block))
    return bytes(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    values = [f32(math.sin(i * .37) * (1 + i // 32) / 19) for i in range(512 * 5)]
    values[:32] = [0.] * 32
    # Exact scale 1 tests ties-to-even, both signs and clamping endpoints.
    values[32:64] = [127., -127.] + [f32((i - 15) + .5) for i in range(30)]
    raw = struct.pack('<' + 'f' * len(values), *values)
    files = {'input.f32': raw, 'expected.q8': encode(values)}
    for name, data in files.items():
        (a.output / name).write_bytes(data)
    (a.output / 'manifest.json').write_text(json.dumps({
        'format': 'q8-block32-fp16-scale-rne-le', 'tokens': 5,
        'oracle': 'Python struct IEEE FP16 and independent scalar round-even',
        'sha256': {n: hashlib.sha256(b).hexdigest() for n, b in files.items()},
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
