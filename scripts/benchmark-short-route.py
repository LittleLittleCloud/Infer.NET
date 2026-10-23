# -*- coding: utf-8 -*-
import argparse,pathlib,json,urllib.request,time,sys,datetime,subprocess,re
p=argparse.ArgumentParser(description="Compare explicit short-prefill-tokens=0 versus 128, with a prestarted deployment. No model restarts.")
p.add_argument('--label',choices=['split','local'],required=True)
p.add_argument('--token-file',type=pathlib.Path,required=True)
p.add_argument('--endpoint',default='http://127.0.0.1:5200')
p.add_argument('--health',default='http://127.0.0.1:5201/health')
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args();label=a.label;out=a.output;out.mkdir(parents=True,exist_ok=False)
key=a.token_file.read_text().strip()
h=json.load(urllib.request.urlopen(a.health,timeout=5));assert h['ready'] and h['state']=='Ready' and h['weightBits']==8 and h['kv']=='fp32',h
(out/'health.json').write_text(json.dumps(h,indent=2));print('OUT',out,flush=True)
system={'role':'system','content':'You are a helpful assistant.'}
cases=[('hello',[{'role':'user','content':'你好'}],32),
('arithmetic',[{'role':'user','content':'17乘以23是多少？只给出数字。'}],32),
('code',[{'role':'user','content':'写一个 C# 函数判断整数是否是偶数，只输出代码。'}],96),
('explanation',[{'role':'user','content':'用两句话解释 prefill 和 decode 的区别。'}],96),
('history',[{'role':'user','content':'记住我的编号是7391。'},{'role':'assistant','content':'好的，你的编号是7391。'},{'role':'user','content':'我的编号是多少？只给出数字。'}],32),
('long-control',[{'role':'user','content':'下面是背景材料：'+('天空是蓝色的，树叶是绿色的。'*40)+'请用一句话总结。'}],32)]
records=[]
for name,messages,maximum in cases:
 expected=None
 for trial in range(4):
  deadline=time.monotonic()+15
  while time.monotonic()<deadline:
   hh=json.load(urllib.request.urlopen(a.health,timeout=5))
   if hh['ready'] and hh['state']=='Ready':break
   time.sleep(.05)
  assert hh['pid']==h['pid'] and hh['ready']
  request={'prompt':'','messages':[system]+messages,'maxTokens':maximum,'model':'qwen3.6-35b-a3b','backend':'metal','profile':True}
  start=time.monotonic();events=[]
  req=urllib.request.Request(a.endpoint.rstrip('/')+'/v1/infer/stream',data=json.dumps(request).encode(),headers={'Content-Type':'application/json','X-DeepInfer-Token':key})
  with urllib.request.urlopen(req,timeout=180) as response:
   for line in response:
    if line.startswith(b'data:'):events.append({'seconds':time.monotonic()-start,'event':json.loads(line[5:])})
  seconds=time.monotonic()-start;(out/(name+'-'+str(trial)+'.json')).write_text(json.dumps({'request':request,'events':events},indent=2,ensure_ascii=False));result=events[-1]['event']['result'];assert result,result
  if expected is None:expected=result['tokenIds']
  assert result['tokenIds']==expected,(name,'nondeterministic')
  path='metal-local' if label=='local' and result['promptTokens']<=128 else 'cuda-metal-split';assert result['executionPath']==path,(path,result['executionPath'])
  item={'case':name,'trial':trial,'warmup':trial==0,'seconds':seconds,'ttft':next(e['seconds'] for e in events if e['event'].get('tokenId') is not None),'result':result}
  records.append(item);(out/(name+'-'+str(trial)+'.json')).write_text(json.dumps({'request':request,'events':events},indent=2,ensure_ascii=False));(out/'summary.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
  print(name,trial,result['promptTokens'],path,round(item['ttft'],4),round(result['timings']['prefillMs'],2),len(result['tokenIds']),flush=True)
print('COMPLETE',out,flush=True)
