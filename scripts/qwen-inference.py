#!/usr/bin/env python3
"""Official Qwen 3B -> BF16 Metal -> independent worker -> coordinator -> CLI acceptance."""
import argparse,json,os,pathlib,signal,socket,subprocess,tempfile,time,urllib.request,urllib.error,uuid
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--executable',type=pathlib.Path,default=ROOT/'src/DeepInfer.Host/bin/Release/net10.0/deepinfer')
p.add_argument('--reference',type=pathlib.Path,default=ROOT/'artifacts/qwen-reference.json')
p.add_argument('--output',type=pathlib.Path,default=ROOT/'artifacts/qwen-e2e.json')
p.add_argument('--skip-verification',action='store_true')
a=p.parse_args();exe=str(a.executable.resolve());directory=ROOT/'models/qwen2.5-3b-instruct'
token=uuid.uuid4().hex;env={**os.environ,'DEEPINFER_TOKEN':token};processes=[];logs=[];checks=0

def check(value,label):
 global checks
 assert value,label
 checks+=1;print('PASS:',label,flush=True)

def port():
 with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def start(*args):
 log=tempfile.TemporaryFile(mode='w+');process=subprocess.Popen([exe,*args],env=env,stdout=log,stderr=log)
 processes.append(process);logs.append(log);return process

def rpc(origin,path,data=None):
 request=urllib.request.Request(origin+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json','X-DeepInfer-Token':token})
 try:
  with urllib.request.urlopen(request,timeout=300) as response:
   data=response.read();return response.status,json.loads(data) if data else None
 except urllib.error.HTTPError as error:
  data=error.read();return error.code,json.loads(data) if data else None

def wait(predicate):
 deadline=time.monotonic()+180
 while time.monotonic()<deadline:
  if any(p.poll() is not None for p in processes):raise AssertionError('A process exited before cluster was ready')
  try:
   result=predicate()
   if result:return result
  except (urllib.error.URLError,ConnectionError,TimeoutError):pass
  time.sleep(.2)
 raise AssertionError('Cluster startup deadline exceeded')
try:
 verification=''
 if not a.skip_verification:
  command=[exe,'verify-qwen','--weights',str(directory),'--max-tokens','48']
  if a.reference.exists():command+=['--reference',str(a.reference)]
  verified=subprocess.run(command,env=env,text=True,capture_output=True,timeout=300)
  print(verified.stdout,flush=True);print(verified.stderr,flush=True)
  check(verified.returncode==0,'BF16 kernel, Qwen tokenizer, full logits and generation verification');verification=verified.stdout
 origin=f'http://127.0.0.1:{port()}';worker=f'http://127.0.0.1:{port()}'
 start('coordinator','--listen',origin);wait(lambda:rpc(origin,'/health')[0]==200)
 start('worker','--id','qwen-worker','--listen',worker,'--coordinator',origin,'--backend','metal','--weights',str(directory),'--model','qwen2.5-3b-instruct')
 snapshot=wait(lambda: (s if len((s:=rpc(origin,'/v1/workers')[1])['workers'])==1 else None))
 model=snapshot['workers'][0]['registration']['models'][0]
 check(model['id']=='qwen2.5-3b-instruct' and model['format']=='hf-qwen2' and not model['isSimulation'],'worker loaded the real Qwen 3B safetensors model')
 request={'prompt':'用一句话介绍你自己。','maxTokens':48,'model':'qwen2.5-3b-instruct','backend':'metal'}
 code,result=rpc(origin,'/v1/infer',request)
 check(code==200 and result['workerId']=='qwen-worker','coordinator routed inference to the independent Qwen worker')
 check(not result['isSimulation'] and result['gpuDispatches']>0 and result['gpuComputeMs']>0,'Qwen result reports actual completed Metal work')
 check(result['finishReason']=='eos' and len(result['tokenIds'])>0 and 'Qwen' in result['text'],'Qwen generated Chinese self-introduction and stopped on EOS')
 if a.reference.exists():
  reference=json.loads(a.reference.read_text());check(result['tokenIds']==reference['generatedTokenIds'],'RPC result matches independent Transformers FP32 greedy token IDs')
 again=rpc(origin,'/v1/infer',request)[1]
 check(again['tokenIds']==result['tokenIds'],'request-local KV cache resets between Qwen requests')
 cli=subprocess.run([exe,'infer','--endpoint',origin,'--json'],env=env,text=True,capture_output=True,timeout=300)
 check(cli.returncode==0 and json.loads(cli.stdout)['tokenIds']==result['tokenIds'],'default CLI now uses Qwen 3B / Metal end to end')
 code,other=rpc(origin,'/v1/infer',{**request,'prompt':'计算2加2，只回答数字。','maxTokens':16})
 check(code==200 and '4' in other['text'] and other['text']!=result['text'],'Qwen follows a different Chinese prompt')
 check(rpc(origin,'/v1/infer',{**request,'prompt':'hello '*2200})[0]==400,'Qwen application context overflow returns HTTP 400')
 check(rpc(origin,'/v1/infer',{**request,'backend':'cuda'})[0]==503,'unimplemented CUDA never substitutes another backend')
 bench=subprocess.run([exe,'bench','--endpoint',origin,'--requests','1','--warmup','0','--max-tokens','16'],env=env,text=True,capture_output=True,timeout=300)
 check(bench.returncode==0,'default benchmark exercises Qwen 3B through real RPC')
 benchmark=json.loads(bench.stdout)
 check(not benchmark['isSimulation'] and benchmark['tokensPerSecond']>0 and benchmark['gpuDispatches']>0,'Qwen benchmark reports real model tokens and GPU commands')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'executable':exe,'checks':checks,'verification':verification,'result':result,'otherResult':other,'benchmark':benchmark},ensure_ascii=False,indent=2)+'\n')
 print(f'{checks} Qwen 3B end-to-end checks passed; evidence: {a.output}',flush=True)
finally:
 for p in reversed(processes):
  if p.poll() is None:
   p.send_signal(signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait()
 if __import__('sys').exc_info()[0]:
  for log in logs:log.seek(0);print(log.read())
 for log in logs:log.close()
