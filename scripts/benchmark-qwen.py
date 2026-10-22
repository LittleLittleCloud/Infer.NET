#!/usr/bin/env python3
"""Comparable single-worker Qwen BF16 benchmarks; no concurrent model replicas."""
import argparse,json,os,pathlib,signal,socket,subprocess,tempfile,time,urllib.request,urllib.error,uuid
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--executable',type=pathlib.Path,default=ROOT/'artifacts/native/osx-arm64/deepinfer')
p.add_argument('--sizes',nargs='+',choices=['3b','7b'],default=['3b','7b'])
p.add_argument('--requests',type=int,default=3)
p.add_argument('--warmup',type=int,default=1)
p.add_argument('--max-tokens',type=int,default=128)
p.add_argument('--output',type=pathlib.Path,default=ROOT/'artifacts/qwen-decode-comparison.json')
a=p.parse_args();exe=str(a.executable.resolve());report=[]
PROMPT='请用中文详细解释什么是机器学习，至少写五百字。'

def port():
 with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
for size in a.sizes:
 processes=[];logs=[];token=uuid.uuid4().hex;env={**os.environ,'DEEPINFER_TOKEN':token}
 def start(*args):
  log=tempfile.TemporaryFile(mode='w+');process=subprocess.Popen([exe,*args],env=env,stdout=log,stderr=log)
  processes.append(process);logs.append(log);return process
 def rpc(path,data=None):
  request=urllib.request.Request(origin+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json','X-DeepInfer-Token':token})
  with urllib.request.urlopen(request,timeout=300) as response:return json.load(response)
 try:
  model=f'qwen2.5-{size}-instruct';origin=f'http://127.0.0.1:{port()}'
  print(f'Starting {model} (one BF16 Metal worker)…',flush=True)
  start('coordinator','--listen',origin)
  worker=start('worker','--listen',f'http://127.0.0.1:{port()}','--id',f'benchmark-{size}','--backend','metal','--weights',str(ROOT/'models'/model),'--model',model,'--coordinator',origin)
  deadline=time.monotonic()+240
  while time.monotonic()<deadline:
   if any(p.poll() is not None for p in processes):raise RuntimeError('A benchmark server process exited')
   try:
    if len(rpc('/v1/workers')['workers'])==1:break
   except (urllib.error.URLError,TimeoutError,ConnectionError):pass
   time.sleep(.2)
  else:raise RuntimeError('Model did not load before deadline')
  result=subprocess.run([exe,'bench','--endpoint',origin,'--model',model,'--backend','metal','--requests',str(a.requests),'--warmup',str(a.warmup),'--max-tokens',str(a.max_tokens),'--prompt',PROMPT],env=env,text=True,capture_output=True,timeout=1800)
  if result.returncode:raise RuntimeError(result.stderr)
  benchmark=json.loads(result.stdout)
  assert not benchmark['isSimulation'] and benchmark['decodeTokens']>0 and benchmark['decodeMs']>0
  assert abs(benchmark['decodeTokensPerSecond']-benchmark['decodeTokens']*1000/benchmark['decodeMs'])<1e-8
  answer=rpc('/v1/infer',{'prompt':'用一句话介绍你自己。','maxTokens':64,'model':model,'backend':'metal'})
  assert not answer['isSimulation'] and answer['gpuDispatches']>0 and len(answer['tokenIds'])>0
  assert ('Qwen' in answer['text'] or '阿里' in answer['text'] or '通义' in answer['text']),answer['text']
  assert answer['timings']['decodeTokens']==max(0,len(answer['tokenIds'])-1)
  row={'model':model,'prompt':PROMPT,'precision':'original BF16 weights / FP32 accumulation','benchmark':benchmark,'answer':answer}
  report.append(row)
  print(f"{size}: decode={benchmark['decodeTokensPerSecond']:.3f} tokens/s; mean prefill={benchmark['meanPrefillMs']:.1f}ms; end-to-end={benchmark['tokensPerSecond']:.3f} tokens/s",flush=True)
  print(answer['text'],flush=True)
  a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps({'executable':exe,'warmup':a.warmup,'requests':a.requests,'maxTokens':a.max_tokens,'rows':report},ensure_ascii=False,indent=2)+'\n')
 finally:
  for process in reversed(processes):
   if process.poll() is None:
    process.send_signal(signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill();process.wait()
  if __import__('sys').exc_info()[0]:
   for log in logs:log.seek(0);print(log.read())
  for log in logs:log.close()
print('Benchmark evidence:',a.output,flush=True)
