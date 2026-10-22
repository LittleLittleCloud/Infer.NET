#!/usr/bin/env python3
"""Controlled own-engine comparison: host executor and complete GPU executors."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--host',type=Path,default=ROOT/'artifacts/native/osx-arm64/deepinfer')
p.add_argument('--base-port',type=int,default=5400)
p.add_argument('--trials',type=int,default=3)
p.add_argument('--max-tokens',type=int,default=128)
p.add_argument('--variants',nargs='+',choices=['host','serial','concurrent','wide','token-prefill','vector','matrix'],default=['host','serial','concurrent','wide'])
p.add_argument('--output-prefix',default='qwen36-gpu')
a=p.parse_args()
if not 1024<=a.base_port<=65000 or a.trials<1 or not 2<=a.max_tokens<=256:p.error('Invalid benchmark parameters')
if not re.fullmatch('[a-z0-9-]+',a.output_prefix):p.error('Invalid output prefix')
for port in (a.base_port,a.base_port+1):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind(('127.0.0.1',port))
out=ROOT/'artifacts/baselines';out.mkdir(parents=True,exist_ok=True)
endpoint=f'http://127.0.0.1:{a.base_port}'
headers={}
if os.environ.get('DEEPINFER_TOKEN'):headers['X-DeepInfer-Token']=os.environ['DEEPINFER_TOKEN']
reports={}
for variant in a.variants:
    env=dict(os.environ)
    for key in ['DEEPINFER_METAL_SERIAL','DEEPINFER_GRAPH_FORCE_ROUTES','DEEPINFER_GRAPH_DEBUG_DIR',
                'DEEPINFER_GRAPH_DEBUG_POSITION','DEEPINFER_METAL_GPU_PROFILE']:env.pop(key,None)
    env.update(DEEPINFER_METAL_GENERIC='0',DEEPINFER_METAL_EXPERT_UNFUSED='0',
               DEEPINFER_METAL_HOST_EXECUTOR='1' if variant=='host' else '0',
               DEEPINFER_METAL_SERIAL_GRAPH='1' if variant=='serial' else '0',
               DEEPINFER_METAL_PREFILL='0' if variant in ['host','token-prefill'] else '1',
               DEEPINFER_METAL_PREFILL_MATRIX='0' if variant=='vector' else '1',
               DEEPINFER_METAL_ROW_TILE='8' if variant=='wide' else '4')
    print('Starting',variant,flush=True)
    with (out/f'{a.output_prefix}-{variant}-cluster.log').open('w') as log:
        cluster=subprocess.Popen([str(a.host.resolve()),'demo','--model','qwen3.6-35b-a3b','--base-port',str(a.base_port),'--stay'],
                                 cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for _ in range(600):
                if cluster.poll() is not None:raise RuntimeError(f'{variant} cluster exited; inspect its log')
                try:
                    with urllib.request.urlopen(urllib.request.Request(endpoint+'/v1/workers',headers=headers),timeout=1) as r:x=json.load(r)
                    if any(w['healthy'] and w['registration']['models'] for w in x['workers']):break
                except OSError:pass
                time.sleep(.2)
            else:raise TimeoutError('Worker startup timed out')
            path=out/f'{a.output_prefix}-{variant}.json'
            subprocess.run([sys.executable,str(ROOT/'scripts/benchmark-qwen36-own.py'),'--endpoint',endpoint,
                            '--trials',str(a.trials),'--max-tokens',str(a.max_tokens),'--expected-submissions-per-forward',
                            '191' if variant=='host' else '1','--prefill-mode','sequential' if variant in ['host','token-prefill'] else 'batched',
                            '--output',str(path)],cwd=ROOT,env=env,check=True)
            reports[variant]=json.loads(path.read_text())
            reports[variant].update(nativeExecutableSha256=hashlib.sha256(a.host.read_bytes()).hexdigest(),
                nativeMetalLibrarySha256=hashlib.sha256((a.host.parent/'libdeepinfer_metal.dylib').read_bytes()).hexdigest(),
                controllerSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                kernelConfiguration={k:env.get(k,'default') for k in ['DEEPINFER_METAL_HOST_EXECUTOR',
                    'DEEPINFER_METAL_SERIAL_GRAPH','DEEPINFER_METAL_ROW_TILE','DEEPINFER_METAL_VECTOR_QUANT','DEEPINFER_METAL_PREFILL','DEEPINFER_METAL_PREFILL_MATRIX','DEEPINFER_METAL_PREFILL_ROWS','DEEPINFER_METAL_PREFILL_TOKEN_TILE','DEEPINFER_METAL_PREFILL_CHUNK','DEEPINFER_METAL_PREFILL_ATTENTION','DEEPINFER_METAL_PREFILL_ATTENTION_FEATURES','DEEPINFER_METAL_PREFILL_EXPERTS','DEEPINFER_METAL_PREFILL_DELTA','DEEPINFER_METAL_PREFILL_GEMM','DEEPINFER_METAL_KV']})
            path.write_text(json.dumps(reports[variant],ensure_ascii=False,indent=2)+'\n')
        finally:
            if cluster.poll() is None:
                cluster.send_signal(signal.SIGTERM)
                try:cluster.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(cluster.pid,signal.SIGKILL);cluster.wait()
first=next(iter(reports.values()))['warmup']['result']['tokenIds']
equal=all(t['result']['tokenIds']==first for r in reports.values() for t in [r['warmup'],*r['trials']])
summary={'completedAtUtc':datetime.now(timezone.utc).isoformat(),'crossVariantTokenIdsExactlyEqual':equal,'variants':{}}
for name,r in reports.items():
    entry={k:r[k] for k in ['decodeTokensPerSecond','meanPrefillMs','meanFirstTokenMs','meanRpcMs']}
    entry.update(gpuDispatchesPerRequest=r['trials'][0]['result']['gpuDispatches'],gpuSubmissionsPerRequest=r['trials'][0]['result']['gpuSubmissions'],
                 meanGpuCommandMs=sum(t['result']['gpuComputeMs'] for t in r['trials'])/len(r['trials']))
    summary['variants'][name]=entry
(out/f'{a.output_prefix}-ab-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
if not equal:raise ValueError('Greedy token IDs differ across executor configurations')
