#!/usr/bin/env python3
"""Benchmark our native quantized kernels on actual checkpoint tensors. Stdlib only."""
import argparse
import ctypes as C
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', type=Path, default=ROOT/'native/DeepInfer.Metal/build/libdeepinfer_metal.dylib')
p.add_argument('--output', type=Path, default=ROOT/'artifacts/kernels/quant-gemv.json')
p.add_argument('--repetitions', type=int, default=30)
p.add_argument('--variants', nargs='+', choices=['generic', 'tiled'], default=['generic', 'tiled'])
a = p.parse_args()
if a.repetitions < 5: p.error('Use at least five repetitions')
lib = C.CDLL(str(a.library.resolve()))
def bind(name, result, *args):
    f = getattr(lib, name); f.restype = result; f.argtypes = list(args); return f
ptr, integer, flt = C.c_void_p, C.c_int, C.c_float
create = bind('di_quant_create', ptr, ptr, integer)
destroy = bind('di_quant_destroy', None, ptr)
matrix_create = bind('di_quant_matrix_create', ptr, ptr, integer, integer, integer, integer, integer, ptr, integer)
matrix_buffer = bind('di_quant_matrix_buffer', ptr, ptr, integer)
matrix_destroy = bind('di_quant_matrix_destroy', None, ptr)
gemv = bind('di_quant_matvec', integer, ptr, ptr, ptr, ptr, ptr, integer, integer, ptr, ptr, integer)
directory = ROOT/'models/qwen3.6-35b-a3b-5bit-mlx'
weight_map = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
headers = {}
def tensor(name):
    file = directory/weight_map[name]
    if file not in headers:
        with file.open('rb') as f:
            length = int.from_bytes(f.read(8),'little'); headers[file] = (8+length, json.loads(f.read(length)))
    offset, header = headers[file]
    return file, offset, header[name]
def upload(matrix, name, part):
    file, offset, t = tensor(name)
    start, end = t['data_offsets']; target = matrix_buffer(matrix, part)
    view = memoryview((C.c_ubyte*(end-start)).from_address(target)).cast('B')
    with file.open('rb', buffering=0) as f:
        f.seek(offset+start); read = 0
        while read < len(view):
            n = f.readinto(view[read:])
            if not n: raise EOFError(name)
            read += n

prefix = 'language_model.model.layers.0.'
cases = [('linear_qkv',prefix+'linear_attn.in_proj_qkv',5,False),
         ('linear_a',prefix+'linear_attn.in_proj_a',5,False),
         ('router',prefix+'mlp.gate',8,False),
         ('shared_gate',prefix+'mlp.shared_expert.gate_proj',5,False),
         ('expert_gate',prefix+'mlp.switch_mlp.gate_proj',5,False),
         ('expert_down',prefix+'mlp.switch_mlp.down_proj',5,True),
         ('classifier','language_model.lm_head',5,False)]
provenance = json.loads((directory/'provenance.json').read_text())
report = {'completedAtUtc': None, 'librarySha256': hashlib.sha256(a.library.read_bytes()).hexdigest(),
          'checkpointRepository': provenance['repository'], 'checkpointRevision': provenance['revision'],
          'purpose': 'Own Metal microbenchmark; repeated warm resident weights, not end-to-end throughput or kernel-only counters',
          'repetitions': a.repetitions, 'warmup': 5, 'cases': []}
for label, name, bits, batched in cases:
    shape = tensor(name+'.weight')[2]['shape']; experts = shape[0] if len(shape)==3 else 1
    rows, columns = shape[-2], shape[-1]*32//bits; count = 8 if experts>1 else 1
    ids = (integer*count)(*[1+32*i for i in range(count)]) if count>1 else (integer*1)(0)
    x = (flt*(columns*(count if batched else 1)))(*[math.sin(i*.017)*.2 for i in range(columns*(count if batched else 1))])
    case = {'name': label, 'tensor': name, 'experts': experts, 'selectedExperts': list(ids),
            'rows': rows, 'columns': columns, 'bits': bits, 'batchedInput': batched, 'variants': {}}
    baseline = None
    for variant in a.variants:
        os.environ['DEEPINFER_METAL_GENERIC'] = '1' if variant=='generic' else '0'
        error = C.create_string_buffer(2048); ctx = create(error, len(error)); matrix = None
        if not ctx: raise RuntimeError(error.value.decode())
        try:
            matrix = matrix_create(ctx, experts, rows, columns, bits, 64, error, len(error))
            if not matrix: raise RuntimeError(error.value.decode())
            for part, suffix in enumerate(['weight','scales','biases']): upload(matrix, name+'.'+suffix,part)
            y = (flt*(rows*count))(); gpu = C.c_double(); samples=[]
            for iteration in range(5+a.repetitions):
                start = time.perf_counter_ns()
                rc = gemv(ctx,matrix,x,y,ids,count,int(batched),C.byref(gpu),error,len(error))
                wall = (time.perf_counter_ns()-start)/1e6
                if rc: raise RuntimeError(error.value.decode())
                if iteration>=5: samples.append({'gpuCommandMs': gpu.value, 'wallMs': wall})
            actual = list(y)
            if not all(math.isfinite(v) for v in actual): raise ValueError('Non-finite output')
            if baseline is None: baseline = actual
            differences = [u-v for u,v in zip(actual,baseline)]
            rel = math.sqrt(sum(d*d for d in differences)/max(sum(v*v for v in baseline),1e-30))
            absolute = max(abs(d) for d in differences)
            if rel>1e-4 or absolute>.01: raise ValueError(f'Output mismatch {label}: {rel}, {absolute}')
            record = {'medianGpuCommandMs': statistics.median(s['gpuCommandMs'] for s in samples),
                      'medianWallMs': statistics.median(s['wallMs'] for s in samples),
                      'relativeL2VsFirstVariant': rel,'maxAbsoluteVsFirstVariant': absolute,'samples':samples}
            case['variants'][variant]=record
            print(label,variant,rows,columns,'GPU ms',record['medianGpuCommandMs'],'wall ms',record['medianWallMs'],flush=True)
        finally:
            if matrix: matrix_destroy(matrix)
            destroy(ctx)
    report['cases'].append(case)
report['completedAtUtc']=datetime.now(timezone.utc).isoformat()
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
