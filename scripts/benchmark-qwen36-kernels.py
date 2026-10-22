#!/usr/bin/env python3
"""Sequential Native AOT A/B: old GEMV, tiled GEMV, and tiled + fused experts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--host', type=Path, default=ROOT/'artifacts/native/osx-arm64/deepinfer')
p.add_argument('--base-port', type=int, default=5300)
p.add_argument('--trials', type=int, default=3)
p.add_argument('--max-tokens', type=int, default=128)
p.add_argument('--variants', nargs='+', choices=['generic-unfused','tiled-unfused','optimized'],
               default=['generic-unfused','tiled-unfused','optimized'])
a = p.parse_args()
if not 1024 <= a.base_port <= 65000 or a.trials < 1 or not 2 <= a.max_tokens <= 256:
    p.error('Invalid port/trial/output range')
for port in (a.base_port,a.base_port+1):
    with socket.socket() as s: s.bind(('127.0.0.1',port))
out = ROOT/'artifacts/baselines'; out.mkdir(parents=True,exist_ok=True)
origin = f'http://127.0.0.1:{a.base_port}'
headers = {}
if os.environ.get('DEEPINFER_TOKEN'): headers['X-DeepInfer-Token']=os.environ['DEEPINFER_TOKEN']
reports = {}
for variant in a.variants:
    env = dict(os.environ)
    env.pop('DEEPINFER_METAL_SERIAL',None)
    env['DEEPINFER_METAL_HOST_EXECUTOR']='1'
    env['DEEPINFER_METAL_GENERIC']='1' if variant=='generic-unfused' else '0'
    env['DEEPINFER_METAL_EXPERT_UNFUSED']='0' if variant=='optimized' else '1'
    print('Starting',variant,flush=True)
    with (out/f'qwen36-kernel-{variant}-cluster.log').open('w') as log:
        cluster = subprocess.Popen([str(a.host.resolve()),'demo','--model','qwen3.6-35b-a3b',
                                    '--base-port',str(a.base_port),'--workers','1','--stay'],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            for _ in range(600):
                if cluster.poll() is not None: raise RuntimeError('Cluster exited; inspect its log')
                try:
                    with urllib.request.urlopen(urllib.request.Request(origin+'/v1/workers',headers=headers),timeout=1) as r:
                        x=json.load(r)
                    if any(w['healthy'] and any(m['id']=='qwen3.6-35b-a3b' for m in w['registration']['models']) for w in x['workers']): break
                except OSError: pass
                time.sleep(.2)
            else: raise TimeoutError('Model startup timed out')
            path=out/f'qwen36-kernel-{variant}.json'
            subprocess.run([sys.executable,str(ROOT/'scripts/benchmark-qwen36-own.py'),
                            '--endpoint',origin,'--trials',str(a.trials),'--max-tokens',str(a.max_tokens),
                            '--expected-submissions-per-forward','191' if variant=='optimized' else '231',
                            '--output',str(path)],cwd=ROOT,env=env,check=True)
            reports[variant]=json.loads(path.read_text())
        finally:
            if cluster.poll() is None:
                cluster.send_signal(signal.SIGTERM)
                try: cluster.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(cluster.pid,signal.SIGKILL);cluster.wait()

first=next(iter(reports.values())); ids=first['warmup']['result']['tokenIds']
equal=all(t['result']['tokenIds']==ids for r in reports.values() for t in [r['warmup'],*r['trials']])
summary={'completedAtUtc':datetime.now(timezone.utc).isoformat(),'crossVariantTokenIdsExactlyEqual':equal,'variants':{}}
for name,r in reports.items():
    summary['variants'][name]={k:r[k] for k in ['decodeTokensPerSecond','meanPrefillMs','meanFirstTokenMs','meanRpcMs']}
    summary['variants'][name].update({'gpuDispatchesPerRequest':r['trials'][0]['result']['gpuDispatches'],
                                    'gpuSubmissionsPerRequest':r['trials'][0]['result']['gpuSubmissions'],
                                    'meanGpuCommandMs':sum(t['result']['gpuComputeMs'] for t in r['trials'])/len(r['trials'])})
(out/'qwen36-kernel-ab-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
if not equal: raise ValueError('Cross-variant greedy IDs differ; inspect the numerical reference before accepting the change')
