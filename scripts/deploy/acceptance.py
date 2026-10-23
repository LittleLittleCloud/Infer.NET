#!/usr/bin/env python3
"""Exercise a running real deployment. Requires no shell/SSH during inference."""
import argparse,concurrent.futures,hashlib,json,pathlib,time,urllib.request,urllib.error
p=argparse.ArgumentParser();p.add_argument('--endpoint',default='http://127.0.0.1:5200');p.add_argument('--worker',default='http://127.0.0.1:5201');p.add_argument('--token-file',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False);token=pathlib.Path(a.token_file).read_text().strip();checks=[]
base={'prompt':'用一句话介绍你自己。','maxTokens':32,'model':'qwen3.6-35b-a3b','backend':'metal','profile':True,'benchmarkIgnoreEos':True}
def request(path,data=None,auth=token,root=a.endpoint):
 req=urllib.request.Request(root+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json','X-DeepInfer-Token':auth})
 try:
  with urllib.request.urlopen(req,timeout=320) as r:return r.status,json.load(r)
 except urllib.error.HTTPError as e:return e.code,json.loads(e.read() or '{}')
def check(name,condition):
 checks.append({'name':name,'pass':bool(condition)});print(('PASS ' if condition else 'FAIL ')+name,flush=True)
 if not condition:raise RuntimeError(name)
started=time.monotonic()
while True:
 try:
  status,ready=request('/ready',root=a.worker)
  workers=request('/v1/workers')[1].get('workers',[])
  if status==200 and any(w['healthy'] for w in workers):break
 except (OSError,ValueError):pass
 if time.monotonic()-started>600:raise TimeoutError('Deployment did not become ready')
 time.sleep(2)
pid=ready['pid'];check('readiness advertises 8bit FP32 2K',ready['weightBits']==8 and ready['kv']=='fp32' and ready['maxContextTokens']==2048)
check('API rejects wrong authentication',request('/v1/infer',base,auth='wrong')[0]==401)
check('output limit rejected',request('/v1/infer',dict(base,maxTokens=257))[0]==400)
check('context overflow rejected without truncation',request('/v1/infer',dict(base,prompt='hello '*2200))[0]==400)
results=[]
for i in range(3):
 status,result=request('/v1/infer',base);check('resident request '+str(i),status==200 and len(result['tokenIds'])==32)
 results.append(result);(out/('request-'+str(i)+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
check('three requests reset state deterministically',all(r['tokenIds']==results[0]['tokenIds'] for r in results))
def stream(cancel=False):
 req=urllib.request.Request(a.endpoint+'/v1/infer/stream',data=json.dumps(base).encode(),headers={'Content-Type':'application/json','X-DeepInfer-Token':token})
 events=[]
 with urllib.request.urlopen(req,timeout=320) as response:
  for raw in response:
   line=raw.decode().strip()
   if not line.startswith('data: '):continue
   event=json.loads(line[6:]);events.append(event)
   if cancel and event.get('tokenId') is not None:return events
 return events
events=stream();final=next(e['result'] for e in events if e.get('result'))
check('streamed token IDs and text match final result',[e['tokenId'] for e in events if e.get('tokenId') is not None]==final['tokenIds'] and ''.join(e.get('text') or '' for e in events)==final['text'])
check('stream and nonstream agree',final['tokenIds']==results[0]['tokenIds'])
check('client can cancel after first streamed token',len(stream(True))==1)
time.sleep(.5)
status,result=request('/v1/infer',base);check('next request correct after cancellation',status==200 and result['tokenIds']==results[0]['tokenIds'])
status,eos=request('/v1/infer',dict(base,prompt='只回答一个字：好。不要解释。',maxTokens=256,benchmarkIgnoreEos=False))
check('natural EOS terminates before limit',status==200 and eos['finishReason']=='eos' and len(eos['tokenIds'])<256)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 busy=list(pool.map(lambda _:request('/v1/infer',dict(base,maxTokens=128)),range(8)))
check('bounded queue provides explicit backpressure',any(code==429 for code,_ in busy) and all(code in (200,429) for code,_ in busy))
status,health=request('/health',root=a.worker);check('Mac model PID remains unchanged',status==200 and health['pid']==pid)
summary={'checks':checks,'pid':pid,'build':health['build'],'completedRequests':health['completedRequests'],'timings':[r['timings'] for r in results],'outputSha256':hashlib.sha256(json.dumps(results[0]['tokenIds']).encode()).hexdigest(),'queueStatuses':[s for s,_ in busy],'elapsedSeconds':time.monotonic()-started}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('Completed',len(checks),'checks; evidence',out,flush=True)
