#!/usr/bin/env python3
"""Compare layer-major CUDA prefill chunks with the 32-token control.

Exclusive-GPU diagnostic. It does not exercise the RPC transfer or Mac suffix.
"""
import argparse
import array
import ctypes as c
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--weights', required=True)
p.add_argument('--library', required=True)
p.add_argument('--output', required=True, type=pathlib.Path)
p.add_argument('--chunk', required=True, type=int, choices=(256, 512, 1024, 2048))
p.add_argument('--tokens', type=int, help='Synthetic token count; defaults to --chunk')
p.add_argument('--first-chunk', type=int, choices=(0, 128), default=0,
               help='Also compare a 128-token first tile followed by --chunk tiles')
p.add_argument('--kv-bytes', type=int, choices=(2, 4), default=4)
p.add_argument('--mode', type=int, choices=(1, 5), default=5)
a = p.parse_args()
token_count = a.tokens or a.chunk
if token_count < 1 or token_count > 262144:
    p.error('--tokens must be 1..262144')
a.output.mkdir(parents=True, exist_ok=False)
tokens = [100 + (i * 137) % 247000 for i in range(token_count)]
weight_hash = hashlib.sha256()
with open(a.weights, 'rb') as stream:
    for block in iter(lambda: stream.read(8 << 20), b''):
        weight_hash.update(block)
assert weight_hash.hexdigest() == 'ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77'

dll = c.CDLL(str(pathlib.Path(a.library).resolve()))
err = c.create_string_buffer(2048)
def bind(name, types, result=c.c_int):
    fn = getattr(dll, 'di_cuda_prefix_' + name)
    fn.argtypes = types
    fn.restype = result
    return fn
def ok(code):
    if code:
        raise RuntimeError(err.value.decode() or 'Native return ' + str(code))
create = bind('create', [c.c_char_p, c.c_int, c.c_int, c.c_char_p, c.c_int], c.c_void_p)
destroy = bind('destroy', [c.c_void_p], None)
reset = bind('reset', [c.c_void_p, c.c_char_p, c.c_int])
mode = bind('execution_mode', [c.c_void_p, c.c_int])
chunk_mode = bind('chunk_mode', [c.c_void_p, c.c_int])
run = bind('run', [c.c_void_p, c.POINTER(c.c_int), c.c_int, c.c_int,
                   c.POINTER(c.c_float), c.POINTER(c.c_float), c.c_char_p, c.c_int])
state = bind('state', [c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_void_p, c.c_int, c.c_char_p, c.c_int])
kv_range = bind('kv_range', [c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int,
                             c.c_void_p, c.c_int, c.c_char_p, c.c_int])
assert bind('abi_version', [])() == 7
handle = create(a.weights.encode(), token_count, a.kv_bytes, err, len(err))
if not handle:
    raise RuntimeError(err.value.decode())
try:
    ok(mode(handle, a.mode))
    ok(chunk_mode(handle, a.chunk))
    def execute(size, first=0):
        ok(reset(handle, err, len(err)))
        hidden = bytearray()
        started = time.perf_counter()
        offsets = []
        offset = 0
        if first:
            offsets.append((0, min(first, len(tokens))))
            offset = min(first, len(tokens))
        while offset < len(tokens):
            count = min(size, len(tokens) - offset)
            offsets.append((offset, count))
            offset += count
        for offset, count in offsets:
            part = tokens[offset:offset + count]
            ids = (c.c_int * len(part))(*part)
            output = (c.c_float * (len(part) * 2048))()
            ok(run(handle, ids, len(part), offset, output, None, err, len(err)))
            hidden.extend(bytes(output))
        seconds = time.perf_counter() - started
        states = []
        for layer in range(20):
            for tensor in range(2):
                size_bytes = (len(tokens) * 512 * a.kv_bytes if (layer + 1) % 4 == 0 else
                              32 * 128 * 128 * 4 if tensor == 0 else 3 * 8192 * 4)
                data = c.create_string_buffer(size_bytes)
                ok(state(handle, layer, tensor, len(tokens), data, size_bytes, err, len(err)))
                states.append(hashlib.sha256(data.raw).hexdigest())
                if size == a.chunk and first == 0 and len(tokens) <= a.chunk and (layer + 1) % 4 == 0:
                    kv = c.create_string_buffer(size_bytes)
                    ok(kv_range(handle, layer, tensor, 0, len(tokens), kv, size_bytes, err, len(err)))
                    if kv.raw != data.raw:
                        raise RuntimeError('Incremental KV differs from final KV')
        floats = array.array('f')
        floats.frombytes(hidden)
        if not all(math.isfinite(value) for value in floats):
            raise RuntimeError('Nonfinite hidden')
        return {'chunk': size, 'firstChunk': first, 'seconds': seconds,
                'hiddenSha256': hashlib.sha256(hidden).hexdigest(), 'stateSha256': states}

    records = []
    schedules = [(32, 0), (a.chunk, 0)]
    if a.first_chunk:
        schedules.append((a.chunk, a.first_chunk))
    for trial in range(4):
        for size, first in (schedules if trial % 2 == 0 else list(reversed(schedules))):
            record = execute(size, first)
            record['trial'] = trial
            record['warmup'] = trial == 0
            records.append(record)
            print(json.dumps({'trial': trial, 'chunk': size, 'firstChunk': first,
                              'seconds': record['seconds']}), flush=True)
    reference = records[0]
    exact = all(r['hiddenSha256'] == reference['hiddenSha256'] and
                r['stateSha256'] == reference['stateSha256'] for r in records)
    result = {
        'kind': 'native-cuda-long-chunk-ab-not-e2e',
        'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'librarySha256': hashlib.sha256(pathlib.Path(a.library).read_bytes()).hexdigest(),
        'bundleSha256': weight_hash.hexdigest(), 'weightBits': 8, 'kvBytes': a.kv_bytes,
        'executionMode': a.mode,
        'tokens': len(tokens), 'tokensSha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        'records': records, 'exactHiddenAndState': exact,
        'medianSeconds': {(f'{size}+{first}' if a.first_chunk else str(size)): statistics.median(
            r['seconds'] for r in records if not r['warmup'] and r['chunk'] == size and
            r['firstChunk'] == first) for size, first in schedules},
        'limitations': ['Synthetic token IDs', 'No Mac suffix, RPC, client time or 128K request',
                        'No trace comparison unless a mismatch is separately investigated']
    }
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'exact': exact, 'medianSeconds': result['medianSeconds']}), flush=True)
    raise SystemExit(0 if exact else 2)
finally:
    destroy(handle)
