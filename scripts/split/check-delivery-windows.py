#!/usr/bin/env python3
"""Compare every prefix trace/hidden/state across FP32 windows on an idle 4090."""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--weights', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
digest = hashlib.sha256()
with a.weights.open('rb') as f:
    for block in iter(lambda: f.read(8 << 20), b''):
        digest.update(block)
if digest.hexdigest() != 'ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77':
    raise ValueError('Pinned prefix bundle mismatch')
dll = c.CDLL(str(a.library.resolve()))
error = c.create_string_buffer(2048)
def bind(name, args, ret=c.c_int):
    f = getattr(dll, 'di_cuda_prefix_' + name)
    f.argtypes, f.restype = args, ret
    return f
create = bind('create', [c.c_char_p, c.c_int, c.c_int, c.c_char_p, c.c_int], c.c_void_p)
destroy = bind('destroy', [c.c_void_p], None)
reset = bind('reset', [c.c_void_p, c.c_char_p, c.c_int])
mode = bind('execution_mode', [c.c_void_p, c.c_int])
window = bind('window', [c.c_void_p, c.c_int])
run = bind('run', [c.c_void_p, c.POINTER(c.c_int), c.c_int, c.c_int, c.POINTER(c.c_float), c.POINTER(c.c_float), c.c_char_p, c.c_int])
state = bind('state', [c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_void_p, c.c_int, c.c_char_p, c.c_int])
kv_range = bind('kv_range', [c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_void_p, c.c_int, c.c_char_p, c.c_int])
if bind('abi_version', [])() != 7:
    raise ValueError('Wrong candidate ABI')
def ok(code):
    if code:
        raise RuntimeError(error.value.decode() or str(code))
records = []
for size in (32, 64, 128):
    handle = create(str(a.weights).encode(), 2048, 4, error, len(error))
    if not handle:
        raise RuntimeError(error.value.decode())
    try:
        ok(mode(handle, 5))
        if window(handle, 33) == 0:
            raise AssertionError('Invalid window accepted')
        ok(window(handle, size))
        for length in (1, 7, 17, 32, 33, 63, 64, 65, 127, 128, 131, 17):
            ok(reset(handle, error, len(error)))
            tokens = [(i * 7919 + 17) % 248320 for i in range(length)]
            all_hidden, all_traces = bytearray(), bytearray()
            for at in range(0, length, size):
                ids = tokens[at:at+size]
                hidden = (c.c_float * (len(ids)*2048))()
                trace = (c.c_float * (len(ids)*20*2048))()
                ok(run(handle, (c.c_int * len(ids))(*ids), len(ids), at, hidden, trace, error, len(error)))
                all_hidden.extend(bytes(hidden))
                all_traces.extend(bytes(trace))
            states = []
            for layer in range(20):
                for kind in range(2):
                    count = length*512*4 if (layer+1)%4 == 0 else (32*128*128*4 if kind == 0 else 3*8192*4)
                    output = c.create_string_buffer(count)
                    ok(state(handle, layer, kind, length, output, count, error, len(error)))
                    if (layer+1)%4 == 0:
                        for at in range(0, length, size):
                            take = min(size, length-at)
                            chunk = c.create_string_buffer(take*512*4)
                            ok(kv_range(handle, layer, kind, at, take, chunk, len(chunk), error, len(error)))
                            if chunk.raw != output.raw[at*512*4:(at+take)*512*4]:
                                raise AssertionError('Batch KV export changed bytes')
                    states.append(hashlib.sha256(output.raw).hexdigest())
            record = {'window': size, 'length': length, 'hidden': hashlib.sha256(all_hidden).hexdigest(),
                      'traces': hashlib.sha256(all_traces).hexdigest(), 'states': states}
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        destroy(handle)
baseline = records[:12]
passed = all(records[arm*12+i]['hidden'] == ref['hidden'] and records[arm*12+i]['traces'] == ref['traces']
             and records[arm*12+i]['states'] == ref['states'] for arm in (1, 2) for i, ref in enumerate(baseline))
result = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
          'dll_sha256': hashlib.sha256(a.library.read_bytes()).hexdigest(), 'kv': 'fp32',
          'all_hidden_traces_and_40_states_byte_equal': passed, 'records': records}
(a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
if not passed:
    raise SystemExit('Window correctness gate failed; retain all evidence')
