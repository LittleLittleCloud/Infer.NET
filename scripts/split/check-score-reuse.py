#!/usr/bin/env python3
"""Compare retained and opt-in QK-reuse CUDA prefix on identical pinned token IDs.

Run only after the deployed CUDA agent is stopped and its PID is verified.
This loads a second 18 GB resident prefix and is not a dual-host benchmark.
"""
import argparse
import ctypes as c
import hashlib
import json
import pathlib
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--weights', required=True, type=pathlib.Path)
p.add_argument('--library', required=True, type=pathlib.Path)
p.add_argument('--output', required=True, type=pathlib.Path)
p.add_argument('--tokens-json', type=pathlib.Path)
p.add_argument('--count', type=int, default=131)
p.add_argument('--kv-bytes', type=int, choices=[2, 4], default=4)
p.add_argument('--irregular', action='store_true')
p.add_argument('--trace', action='store_true')
a = p.parse_args()
if not 1 <= a.count <= 262144:
    p.error('count must be 1..262144')
if a.trace and a.count > 1024:
    p.error('full layer trace is limited to 1024 tokens')
a.output.mkdir(parents=True, exist_ok=False)
if a.tokens_json:
    tokens = json.loads(a.tokens_json.read_text())
    if not isinstance(tokens, list) or len(tokens) < a.count:
        p.error('tokens-json must contain at least count token IDs')
    tokens = tokens[:a.count]
else:
    tokens = [100 + (i * 137) % 247000 for i in range(a.count)]
if any(type(t) is not int or not 0 <= t < 248320 for t in tokens):
    p.error('token ID outside pinned vocabulary')

def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

metadata = {
    'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'librarySha256': sha_file(a.library), 'bundleSha256': sha_file(a.weights),
    'tokensSha256': hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
    'inputTokens': a.count, 'kvBytes': a.kv_bytes, 'trace': a.trace,
    'chunkPattern': [1, 7, 17, 32] if a.irregular else [32],
    'kind': 'native-CUDA-front20-only; excludes network, Metal suffix and decode'
}
expected = 'ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77'
if metadata['bundleSha256'] != expected:
    raise RuntimeError('Pinned 8bit prefix bundle SHA256 mismatch')

dll = c.CDLL(str(a.library.resolve()))
error = c.create_string_buffer(2048)
def bind(name, args, ret=c.c_int):
    fn = getattr(dll, 'di_cuda_prefix_' + name)
    fn.argtypes = args
    fn.restype = ret
    return fn
def ok(code):
    if code:
        raise RuntimeError(error.value.decode() or f'CUDA error {code}')

create = bind('create', [c.c_char_p, c.c_int, c.c_int, c.c_char_p, c.c_int], c.c_void_p)
destroy = bind('destroy', [c.c_void_p], None)
reset = bind('reset', [c.c_void_p, c.c_char_p, c.c_int])
mode = bind('execution_mode', [c.c_void_p, c.c_int])
run = bind('run', [c.c_void_p, c.POINTER(c.c_int), c.c_int, c.c_int,
                   c.POINTER(c.c_float), c.POINTER(c.c_float), c.c_char_p, c.c_int])
state = bind('state', [c.c_void_p, c.c_int, c.c_int, c.c_int,
                       c.c_void_p, c.c_int, c.c_char_p, c.c_int])
assert bind('abi_version', [])() == 6
handle = create(str(a.weights.resolve()).encode(), a.count, a.kv_bytes, error, len(error))
if not handle:
    raise RuntimeError(error.value.decode())
records = []
try:
    for execution_mode in (5, 6):
        ok(reset(handle, error, len(error)))
        ok(mode(handle, execution_mode))
        hidden_hash = hashlib.sha256()
        trace_hash = hashlib.sha256()
        chunks = []
        position = 0
        index = 0
        start = time.perf_counter()
        while position < a.count:
            n = min(metadata['chunkPattern'][index % len(metadata['chunkPattern'])], a.count - position)
            ids = (c.c_int * n)(*tokens[position:position+n])
            hidden = (c.c_float * (n * 2048))()
            trace = (c.c_float * (n * 20 * 2048))() if a.trace else None
            ok(run(handle, ids, n, position, hidden, trace, error, len(error)))
            payload = bytes(hidden)
            hidden_hash.update(payload)
            chunks.append(hashlib.sha256(payload).hexdigest())
            if trace is not None:
                trace_hash.update(bytes(trace))
            position += n
            index += 1
        prefix_seconds = time.perf_counter() - start
        state_hashes = []
        start = time.perf_counter()
        for layer in range(20):
            for tensor in range(2):
                size = (a.count * 512 * a.kv_bytes if (layer + 1) % 4 == 0 else
                        32 * 128 * 128 * 4 if tensor == 0 else 3 * 8192 * 4)
                buffer = c.create_string_buffer(size)
                ok(state(handle, layer, tensor, a.count, buffer, size, error, len(error)))
                state_hashes.append(hashlib.sha256(buffer.raw).hexdigest())
        record = {'mode': execution_mode, 'prefixSeconds': prefix_seconds,
                  'stateExportSeconds': time.perf_counter() - start,
                  'hiddenSha256': hidden_hash.hexdigest(), 'chunkSha256': chunks,
                  'stateSha256': state_hashes, 'traceSha256': trace_hash.hexdigest() if a.trace else None}
        records.append(record)
        print(json.dumps({'mode': execution_mode, 'prefixSeconds': prefix_seconds,
                          'stateExportSeconds': record['stateExportSeconds'],
                          'hiddenSha256': record['hiddenSha256'],
                          'traceSha256': record['traceSha256']}), flush=True)
    same = (records[0]['chunkSha256'] == records[1]['chunkSha256'] and
            records[0]['stateSha256'] == records[1]['stateSha256'] and
            records[0]['traceSha256'] == records[1]['traceSha256'])
    result = {**metadata, 'records': records, 'bitwiseEqual': same}
    (a.output / 'result.json').write_text(json.dumps(result, indent=2))
    if not same:
        raise RuntimeError('QK reuse changed hidden, layer trace or final state; see result.json')
    print('PASS: all hidden chunks, optional layer trace and 40 final states match bitwise')
finally:
    destroy(handle)
