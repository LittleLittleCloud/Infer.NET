#!/usr/bin/env python3
"""Diagnostic full-prefix BF16-vs-FP32 run; requires the deployed CUDA agent to be stopped.

This checks deterministic execution and records per-layer drift. It is not a
cross-backend correctness verdict or an end-to-end performance benchmark.
"""
import argparse
import array
import ctypes as c
import hashlib
import json
import math
import os
import pathlib
import subprocess
import time

p=argparse.ArgumentParser()
p.add_argument('--weights',required=True)
p.add_argument('--library',required=True)
p.add_argument('--output',required=True,type=pathlib.Path)
p.add_argument('--tokens',type=int,choices=(128,131),default=131)
p.add_argument('--save-trace',action='store_true',help='Save the 20-layer CUDA trace for offline numerical comparison')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
tokens=[100+(i*137)%247000 for i in range(a.tokens)]
weight_sha=hashlib.sha256()
with open(a.weights,'rb') as f:
    for block in iter(lambda:f.read(8<<20),b''):weight_sha.update(block)
assert weight_sha.hexdigest()=='ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77'
dll=c.CDLL(str(pathlib.Path(a.library).resolve()))
err=c.create_string_buffer(2048)
def bind(name,args,ret=c.c_int):
    fn=getattr(dll,'di_cuda_prefix_'+name);fn.argtypes=args;fn.restype=ret;return fn
def ok(result):
    if result:raise RuntimeError(err.value.decode() or 'Native return '+str(result))
create=bind('create',[c.c_char_p,c.c_int,c.c_int,c.c_char_p,c.c_int],c.c_void_p)
destroy=bind('destroy',[c.c_void_p],None)
reset=bind('reset',[c.c_void_p,c.c_char_p,c.c_int])
mode=bind('execution_mode',[c.c_void_p,c.c_int])
run=bind('run',[c.c_void_p,c.POINTER(c.c_int),c.c_int,c.c_int,c.POINTER(c.c_float),c.POINTER(c.c_float),c.c_char_p,c.c_int])
state=bind('state',[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_int,c.c_char_p,c.c_int])
assert bind('abi_version',[])()==6

def execute(handle,mode_id):
    ok(reset(handle,err,len(err)));ok(mode(handle,mode_id))
    hidden=bytearray();trace=bytearray();start=time.perf_counter()
    for pos in range(0,len(tokens),32):
        part=tokens[pos:pos+32]
        ids=(c.c_int*len(part))(*part)
        out=(c.c_float*(len(part)*2048))()
        layer=(c.c_float*(len(part)*20*2048))()
        ok(run(handle,ids,len(part),pos,out,layer,err,len(err)))
        hidden.extend(bytes(out));trace.extend(bytes(layer))
    seconds=time.perf_counter()-start
    states=[]
    for layer in range(20):
        for kind in range(2):
            size=(len(tokens)*512*2 if (layer+1)%4==0 else
                  32*128*128*4 if kind==0 else 3*8192*4)
            data=c.create_string_buffer(size)
            ok(state(handle,layer,kind,len(tokens),data,size,err,len(err)))
            states.append(hashlib.sha256(data.raw).hexdigest())
            if a.save_trace and mode_id==6:
                path=a.output/f'candidate-state-{layer}-{kind}.bin'
                if not path.exists():path.write_bytes(data.raw)
    values=array.array('f');values.frombytes(trace)
    if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite layer trace')
    if a.save_trace and mode_id==5:
        (a.output/'baseline-trace.f32').write_bytes(trace)
    return {'mode':mode_id,'seconds':seconds,'hiddenSha256':hashlib.sha256(hidden).hexdigest(),
            'traceSha256':hashlib.sha256(trace).hexdigest(),'stateSha256':states},values

handle=create(a.weights.encode(),256,2,err,len(err))
if not handle:raise RuntimeError(err.value.decode())
try:
    baseline,reference=execute(handle,5)
    candidate,observed=execute(handle,6)
    repeat,_=execute(handle,6)
    if any(candidate[k]!=repeat[k] for k in ('hiddenSha256','traceSha256','stateSha256')):
        raise RuntimeError('BF16 full-prefix execution is not deterministic')
    if a.save_trace:
        (a.output/'candidate-trace.f32').write_bytes(observed.tobytes())
    layers=[]
    for layer in range(20):
        diff2=ref2=max_abs=0.0
        for token in range(len(tokens)):
            offset=(token*20+layer)*2048
            for i in range(offset,offset+2048):
                delta=float(observed[i])-float(reference[i])
                diff2+=delta*delta;ref2+=float(reference[i])*float(reference[i])
                max_abs=max(max_abs,abs(delta))
        layers.append({'layer':layer,'relativeL2':math.sqrt(diff2/max(ref2,1e-30)),
                       'maxAbs':max_abs})
    result={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'librarySha256':hashlib.sha256(pathlib.Path(a.library).read_bytes()).hexdigest(),
            'bundleSha256':weight_sha.hexdigest(),'weightBits':8,'kv':'bf16',
            'candidateRouterProjection':'fp32-diagnostic' if os.environ.get('DEEPINFER_CUDA_ROUTER_FP32')=='1' else 'bf16',
            'candidateSharedProjection':'fp32-diagnostic' if os.environ.get('DEEPINFER_CUDA_SHARED_FP32')=='1' else 'bf16',
            'tokens':len(tokens),'tokensSha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            'baseline':baseline,'candidate':candidate,'repeat':repeat,'layerDrift':layers,
            'passed':'finite-and-deterministic-only',
            'limitations':['Synthetic token IDs','No independent Metal reference',
                           'No transfer, suffix, decode or client timing',
                           'Trace readbacks distort performance']}
    (a.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'source':result['source'],'baselineSeconds':baseline['seconds'],
                      'candidateSeconds':candidate['seconds'],'maxLayerRelativeL2':max(x['relativeL2'] for x in layers),
                      'maxLayerAbs':max(x['maxAbs'] for x in layers),'passed':result['passed']}),flush=True)
finally:destroy(handle)
