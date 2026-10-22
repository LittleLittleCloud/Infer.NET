#!/usr/bin/env python3
"""Diagnostic split-command GPU intervals; not an end-to-end throughput benchmark."""
import argparse,collections,hashlib,json,os,signal,socket,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trials',type=int,default=1)
p.add_argument('--prefix',default='qwen36-prefill-profile')
p.add_argument('--base-port',type=int,default=5100)
a=p.parse_args()
assert a.trials>0 and a.prefix.replace('-','').isalnum()
for port in [a.base_port,a.base_port+1]:
 with socket.socket() as sock:sock.bind(('127.0.0.1',port))
out=ROOT/'artifacts/baselines';out.mkdir(exist_ok=True)
logpath=out/(a.prefix+'.log');env={**os.environ,'DEEPINFER_METAL_PREFILL_PROFILE':'1','DEEPINFER_METAL_PREFILL_GEMM':'bf16-nax','DEEPINFER_METAL_KV':'bf16'}
endpoint=f'http://127.0.0.1:{a.base_port}'
headers={'Content-Type':'application/json'}
if os.environ.get('DEEPINFER_TOKEN'):headers['X-DeepInfer-Token']=os.environ['DEEPINFER_TOKEN']
results=[]
with logpath.open('w') as log:
 service=subprocess.Popen([str(ROOT/'artifacts/native/osx-arm64/deepinfer'),'demo','--model','qwen3.6-35b-a3b','--base-port',str(a.base_port),'--stay'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:
  for _ in range(600):
   if service.poll() is not None:raise RuntimeError('Profile worker failed')
   try:
    with urllib.request.urlopen(urllib.request.Request(endpoint+'/v1/workers',headers=headers),timeout=1) as r:j=json.load(r)
    if any(w['healthy'] and w['registration']['models'] for w in j['workers']):break
   except OSError:pass
   time.sleep(.2)
  else:raise RuntimeError('Profile startup timeout')
  prompt=(out/'qwen36-context-fixed-data/8192-4096-prompt.txt').read_text()
  baseline=json.loads((out/'qwen36-bf16-release8k-data/8192-128-trial1.json').read_text())
  for _ in range(a.trials):
   req=urllib.request.Request(endpoint+'/v1/infer',json.dumps(dict(prompt=prompt,maxTokens=2,model='qwen3.6-35b-a3b',backend='metal')).encode(),headers)
   with urllib.request.urlopen(req,timeout=600) as r:result=json.load(r)
   assert result['promptTokens']==8192 and result['tokenIds']==baseline['tokenIds'][:2],result
   results.append(result)
 finally:
  if service.poll() is None:
   service.send_signal(signal.SIGTERM)
   try:service.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(service.pid,signal.SIGKILL);service.wait()
records=[]
for line in logpath.read_text().splitlines():
 if 'PREFILL_STAGE ' in line:records.append(json.loads(line.split('PREFILL_STAGE ',1)[1]))
assert records and len([r for r in records if r['stage']=='embedding-norm'])==a.trials
stages=collections.defaultdict(float)
for r in records:stages[r['stage']]+=r['gpuMs']/a.trials
total=sum(stages.values())
summary=dict(purpose=__doc__,tokensInProfiledChunk=2048,chunkPosition=4096,trials=a.trials,
             warning='Explicit command boundaries/CPU waits perturb scheduling and cache residency. Never use this diagnostic as a production throughput result.',
             stages=[dict(stage=k,gpuMs=v,percent=v/total*100) for k,v in sorted(stages.items(),key=lambda x:-x[1])],
             summedIntervalMs=total,results=results,records=records)
(out/(a.prefix+'.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['stages','summedIntervalMs']},indent=2))
