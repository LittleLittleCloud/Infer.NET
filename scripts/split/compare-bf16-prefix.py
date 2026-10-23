#!/usr/bin/env python3
"""Compare isolated Metal and CUDA 128-token BF16 prefix traces and state.

Offline numerical comparison only. Copying diagnostic artifacts is not the
product's state-transfer path and these timings are not throughput evidence.
"""
import argparse
import array
import hashlib
import json
import math
import pathlib
import struct

p=argparse.ArgumentParser()
p.add_argument('--metal',required=True,type=pathlib.Path)
p.add_argument('--cuda',required=True,type=pathlib.Path)
p.add_argument('--output',required=True,type=pathlib.Path)
p.add_argument('--metal-fp32',action='store_true',help='Compare BF16 CUDA with an independent Metal FP32 control')
a=p.parse_args()
if a.output.exists():raise SystemExit('Use a fresh output file')
metal=json.loads((a.metal/'result.json').read_text())
cuda=json.loads((a.cuda/'result.json').read_text())
tokens=metal['tokens']
assert tokens==cuda['tokens']==128 and cuda['kv'] in ('bf16','fp32')
if a.metal_fp32:
    assert metal['kv']=='fp32' and metal['gemm']=='fp32' and metal['routerProjection']=='fp32', 'Metal FP32 control profile mismatch'
    assert cuda.get('candidateRouterProjection')=='fp32-diagnostic' and cuda.get('candidateSharedProjection')=='fp32-diagnostic', 'CUDA diagnostic profile mismatch'
else:
    assert metal['kv']=='bf16' and metal.get('routerProjection','bf16')==cuda.get('candidateRouterProjection','bf16'),'Router profile mismatch'
expected_tokens=[100+(i*137)%247000 for i in range(tokens)]
assert json.loads((a.metal/'tokens.json').read_text())==expected_tokens
assert cuda['tokensSha256']==hashlib.sha256(json.dumps(expected_tokens).encode()).hexdigest()
def values(path,bf16=False):
    data=path.read_bytes()
    if bf16:
        words=array.array('H');words.frombytes(data)
        return [struct.unpack('f',struct.pack('I',int(v)<<16))[0] for v in words]
    floats=array.array('f');floats.frombytes(data)
    return floats
def compare(name,expected,observed):
    assert len(expected)==len(observed),(name,len(expected),len(observed))
    diff2=ref2=max_abs=0.0
    for x,y in zip(expected,observed):
        if not math.isfinite(x) or not math.isfinite(y):
            return {'name':name,'finite':False,'pass':False}
        delta=float(y)-float(x)
        diff2+=delta*delta;ref2+=float(x)*float(x)
        max_abs=max(max_abs,abs(delta))
    relative=math.sqrt(diff2/max(ref2,1e-30))
    return {'name':name,'finite':True,'relativeL2':relative,'maxAbs':max_abs,
            'pass':relative<1e-4 and max_abs<.01}
ref=values(a.metal/'trace.f32');actual=values(a.cuda/'candidate-trace.f32')
assert len(ref)==len(actual)==tokens*20*2048
checks=[]
for layer in range(20):
    e=array.array('f');o=array.array('f')
    for token in range(tokens):
        start=(token*20+layer)*2048
        e.extend(ref[start:start+2048]);o.extend(actual[start:start+2048])
    checks.append(compare(f'layer-{layer}',e,o))
for layer in range(20):
    for kind in range(2):
        metal_bf16=(layer+1)%4==0 and not a.metal_fp32
        cuda_bf16=(layer+1)%4==0 and cuda['kv']=='bf16'
        checks.append(compare(f'state-{layer}-{kind}',
            values(a.metal/f'state-{layer}-{kind}.bin',metal_bf16),
            values(a.cuda/f'candidate-state-{layer}-{kind}.bin',cuda_bf16)))
result={'kind':'isolated-bf16-prefix-versus-fp32-control-not-e2e' if a.metal_fp32 else 'isolated-bf16-prefix-cross-backend-numerical-check-not-e2e',
        'inputTokens':tokens,'weightBits':8,'referenceKv':metal['kv'],'candidateKv':cuda['kv'],
        'metalTraceSha256':hashlib.sha256((a.metal/'trace.f32').read_bytes()).hexdigest(),
        'cudaTraceSha256':hashlib.sha256((a.cuda/'candidate-trace.f32').read_bytes()).hexdigest(),
        'checks':checks,'passed':all(x['pass'] for x in checks),
        'limitations':['Metal FP32 is an independent device control, not the full-model CPU reference' if a.metal_fp32 else 'Metal BF16 is an independent backend control',
                       'Artifact comparison after separate runs, not product transfer',
                       'No suffix, logits, greedy output, 128K request, or latency gate']}
if a.metal_fp32:
    baseline_path=a.cuda/'baseline-trace.f32'
    baseline=values(baseline_path)
    assert len(baseline)==len(ref)
    control=[]
    for layer in range(20):
        e=array.array('f');o=array.array('f')
        for token in range(tokens):
            start=(token*20+layer)*2048
            e.extend(ref[start:start+2048]);o.extend(baseline[start:start+2048])
        control.append(compare(f'mode5-control-layer-{layer}',e,o))
    result['cudaMode5TraceSha256']=hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    control_key='fp32ControlLayerChecks' if cuda['kv']=='fp32' else 'mixedKvControlLayerChecks'
    result[control_key]=control
    result['controlProfile']='CUDA FP32 GEMM / '+cuda['kv'].upper()+' KV versus Metal FP32 GEMM / FP32 KV'
    if cuda['kv']=='bf16':
        result['limitations'].append('CUDA mode 5 retains BF16 KV; later-layer drift cannot be attributed to a pure FP32 cross-device mismatch')
    if cuda['kv']=='fp32':
        state_control=[]
        for layer in range(20):
            for kind in range(2):
                state_control.append(compare(f'fp32-control-state-{layer}-{kind}',
                    values(a.metal/f'state-{layer}-{kind}.bin'),
                    values(a.cuda/f'baseline-state-{layer}-{kind}.bin')))
        result['fp32ControlStateChecks']=state_control
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'passed':result['passed'],
                  'firstFailed':next((x for x in checks if not x['pass']),None),
                  'worstRelative':max((x for x in checks if x['finite']),key=lambda x:x['relativeL2'])}))
raise SystemExit(0 if result['passed'] else 1)
