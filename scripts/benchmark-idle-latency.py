import argparse,json,urllib.request,pathlib,time,subprocess,re,datetime,hashlib,sys
p=argparse.ArgumentParser(description="Serial idle-latency probe; never starts or restarts a model. Set residency explicitly in the deployment before each run.")
p.add_argument('--label',required=True,choices=['on','off'])
p.add_argument('--token-file',required=True,type=pathlib.Path)
p.add_argument('--endpoint',default='http://127.0.0.1:5200')
p.add_argument('--health',default='http://127.0.0.1:5201/health')
p.add_argument('--worker-log',required=True,type=pathlib.Path)
p.add_argument('--output',required=True,type=pathlib.Path)
p.add_argument('--idle-seconds',default='0,2,10,60')
p.add_argument('--rounds',type=int,default=3)
a=p.parse_args();idle_values=[float(x) for x in a.idle_seconds.split(',')]
assert 1<=a.rounds<=10 and all(0<=x<=600 for x in idle_values)
root=pathlib.Path(__file__).resolve().parents[1];label=a.label;out=a.output;out.mkdir(parents=True,exist_ok=False)
print('OUT',out,flush=True)
key=a.token_file.read_text().strip()
def health():return json.load(urllib.request.urlopen(a.health,timeout=5))
def vm():
 s=subprocess.check_output(['vm_stat'],text=True)
 return {m.group(1).strip('"'):int(m.group(2)) for m in re.finditer(r'^([^:]+):\s+(\d+)\.',s,re.M)}
h=health();assert h['ready'] and h['state']=='Ready',h;pid=h['pid']
(out/'probe.json').write_text(json.dumps({'label':label,'rounds':a.rounds,'idleSeconds':idle_values,'scriptSha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'vmStatPageBytes':int(re.search(r'page size of (\d+)',subprocess.check_output(['vm_stat'],text=True)).group(1))},indent=2));(out/'health-before.json').write_text(json.dumps(h,indent=2));(out/'power.txt').write_text(subprocess.check_output(['pmset','-g','batt'],text=True))
log=a.worker_log;logoffset=log.stat().st_size
payload={'prompt':'','messages':[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':'你好'}],'maxTokens':32,'model':'qwen3.6-35b-a3b','backend':'metal','profile':True}
(out/'request.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
expected=[109266,6115,116577,101882,96263,1710,110450,98691,111454,96598,97319,98179,10992]
# First request is warmup only. Every idle duration has three measured samples.
results=[]
for run,idle in enumerate([0]+idle_values*a.rounds):
 time.sleep(idle)
 assert health()['ready'] and health()['pid']==pid
 before=vm();start=time.monotonic();events=[];error=None
 try:
  req=urllib.request.Request(a.endpoint.rstrip('/')+'/v1/infer/stream',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-DeepInfer-Token':key})
  with urllib.request.urlopen(req,timeout=240) as r:
   for raw in r:
    line=raw.decode().strip()
    if line.startswith('data:'):events.append({'seconds':time.monotonic()-start,'event':json.loads(line[5:])})
 except Exception as ex:error=repr(ex)
 duration=time.monotonic()-start;after=vm();delta={k:after[k]-before[k] for k in ['Swapins','Swapouts','Decompressions','Compressions','Pageins']}
 tokens=[e['event']['tokenId'] for e in events if e['event'].get('tokenId') is not None];tokenEvents=[e for e in events if e['event'].get('tokenId') is not None]
 result={'run':run,'warmup':run==0,'idleSeconds':idle,'seconds':duration,'deltaPages':delta,'events':events,'error':error,'tokensEqual':tokens==expected,'ttft':tokenEvents[0]['seconds'] if tokenEvents else None,'firstDecodeGap':tokenEvents[1]['seconds']-tokenEvents[0]['seconds'] if len(tokenEvents)>1 else None}
 results.append(result);(out/f'run-{run}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));(out/'summary.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='events'} for r in results],indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)
 with log.open() as f:f.seek(logoffset);(out/'decode.log').write_text(f.read())
 if error or tokens!=expected:raise SystemExit('Request failed or token mismatch; preserve evidence and stop')
 deadline=time.monotonic()+10
 while health()['state']!='Ready' and time.monotonic()<deadline:time.sleep(.05)
(out/'health-after.json').write_text(json.dumps(health(),indent=2))
print('COMPLETE',out,flush=True)
