#!/usr/bin/env python3
"""Real weights -> C# tokenizer/transformer -> Metal GPU -> independent workers -> coordinator -> CLI."""
import argparse, json, os, pathlib, signal, socket, subprocess, tempfile, time, urllib.request, urllib.error, uuid
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--executable',type=pathlib.Path,default=ROOT/'src/DeepInfer.Host/bin/Release/net10.0/deepinfer')
p.add_argument('--output',type=pathlib.Path,default=ROOT/'artifacts/real-e2e.json')
p.add_argument('--skip-verification',action='store_true')
a=p.parse_args();exe=str(a.executable.resolve())
weights=str(ROOT/'models/stories15M.bin');tokenizer=str(ROOT/'models/tokenizer.bin')
token=uuid.uuid4().hex;env={**os.environ,'DEEPINFER_TOKEN':token}
processes=[];logs=[];checks=0

def check(value,label):
    global checks
    assert value,label
    checks+=1;print('PASS:',label,flush=True)

def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def start(*args):
    log=tempfile.TemporaryFile(mode='w+')
    process=subprocess.Popen([exe,*args],env=env,stdout=log,stderr=log)
    processes.append(process);logs.append(log);return process

def rpc(origin,path,data=None):
    request=urllib.request.Request(origin+path,data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type':'application/json','X-DeepInfer-Token':token})
    try:
        with urllib.request.urlopen(request,timeout=20) as response:
            data=response.read();return response.status,json.loads(data) if data else None
    except urllib.error.HTTPError as error:
        data=error.read();return error.code,json.loads(data) if data else None

def wait_for(predicate):
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        if any(p.poll() is not None for p in processes):raise AssertionError('A server process exited')
        try:
            value=predicate()
            if value:return value
        except (urllib.error.URLError,TimeoutError,ConnectionError):pass
        time.sleep(.1)
    raise AssertionError('Cluster startup deadline exceeded')

try:
    subprocess.run(['python3',str(ROOT/'scripts/download-model.py')],check=True)
    verification=''
    if not a.skip_verification:
        result=subprocess.run([exe,'verify-model','--weights',weights,'--tokenizer',tokenizer],env=env,text=True,capture_output=True,timeout=90)
        print(result.stdout,flush=True);check(result.returncode==0,'Metal kernel, complete logits and 48-token CPU equivalence')
        verification=result.stdout
    origin=f'http://127.0.0.1:{port()}'
    start('coordinator','--listen',origin)
    wait_for(lambda:rpc(origin,'/health')[0]==200)
    for id,backend in [('metal-one','metal'),('metal-two','metal'),('cpu-reference','cpu')]:
        start('worker','--id',id,'--listen',f'http://127.0.0.1:{port()}','--coordinator',origin,
            '--backend',backend,'--weights',weights,'--tokenizer',tokenizer,'--model','stories15m')
    snapshot=wait_for(lambda: (s if len((s:=rpc(origin,'/v1/workers')[1])['workers'])==3 else None))
    check(all(w['healthy'] for w in snapshot['workers']),'three independent real-model workers are registered')
    check(all(w['registration']['models'][0]['id']=='stories15m' and not w['registration']['models'][0]['isSimulation'] for w in snapshot['workers']),'loaded model capabilities are real, without path leakage')
    request={'prompt':'Once upon a time','maxTokens':48,'model':'stories15m','backend':'metal'}
    results=[]
    for _ in range(4):
        status,result=rpc(origin,'/v1/infer',request);check(status==200,'coordinator routes real Metal generation');results.append(result)
    first=results[0]
    check([r['workerId'] for r in results]==['metal-one','metal-two']*2,'whole requests alternate across independent Metal workers')
    check(all(not r['isSimulation'] and len(r['tokenIds'])==48 and r['gpuDispatches']>0 and r['gpuComputeMs']>0 for r in results),'48 real model tokens and actual GPU dispatch/timing recorded')
    check(all(r['tokenIds']==first['tokenIds'] and r['text']==first['text'] for r in results),'request-local KV caches yield deterministic repeated output')
    check('little girl named Lily' in first['text'] and '[SIMULATION]' not in first['text'],'known pretrained checkpoint produces the expected English continuation')
    status,cpu=rpc(origin,'/v1/infer',{**request,'backend':'cpu'})
    check(status==200 and cpu['tokenIds']==first['tokenIds'] and cpu['gpuDispatches']==0,'CPU worker produces identical token IDs and reports zero GPU work')
    status,other=rpc(origin,'/v1/infer',{**request,'prompt':'The little boy'})
    check(status==200 and other['text']!=first['text'],'changing prompt changes the model continuation')
    check(rpc(origin,'/v1/infer',{**request,'prompt':'hello '*300})[0]==400,'tokenized context overflow is forwarded as HTTP 400')
    check(rpc(origin,'/v1/infer',{**request,'backend':'cuda'})[0]==503,'unsupported CUDA never silently falls back to CPU/Metal')
    cli=subprocess.run([exe,'infer','--endpoint',origin,'--backend','metal','--model','stories15m','--prompt',request['prompt'],'--max-tokens','48','--json'],env=env,text=True,capture_output=True,timeout=20)
    check(cli.returncode==0 and json.loads(cli.stdout)['tokenIds']==first['tokenIds'],'CLI -> coordinator -> independent worker -> Metal -> real text')
    benchmark=subprocess.run([exe,'bench','--endpoint',origin,'--backend','metal','--model','stories15m','--requests','3','--warmup','1','--max-tokens','32'],env=env,text=True,capture_output=True,timeout=30)
    check(benchmark.returncode==0,'real-model benchmark command succeeds')
    b=json.loads(benchmark.stdout)
    check(not b['isSimulation'] and b['outputUnits']==96 and b['tokensPerSecond']>0 and b['syntheticUnitsPerSecond']==0 and b['gpuDispatches']>0,'benchmark reports actual generated tokens, not synthetic units')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'executable':exe,'checks':checks,'verification':verification,'result':first,'benchmark':b},ensure_ascii=False,indent=2)+'\n')
    print(f'{checks} real-inference checks passed; evidence: {a.output}',flush=True)
finally:
    for p in reversed(processes):
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:p.kill();p.wait()
    if __import__('sys').exc_info()[0]:
        for log in logs:log.seek(0);print(log.read())
    for log in logs:log.close()
