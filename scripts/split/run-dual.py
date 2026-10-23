#!/usr/bin/env python3
"""Control both hosts, with model data on authenticated TCP, never SSH.
Requires clean identical commits and already-built binaries/model on both hosts.
"""
import argparse,base64,datetime,hashlib,json,os,pathlib,secrets,socket,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--ssh',default='infer-windows');p.add_argument('--prompt',default='用一句话介绍你自己。');p.add_argument('--max-tokens',type=int,default=8);p.add_argument('--verify',action='store_true');p.add_argument('--run-id');p.add_argument('--port',type=int,default=5210)
a=p.parse_args();root=pathlib.Path(__file__).resolve().parents[2];os.chdir(root)
run=a.run_id or 'split-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
assert all(c.isalnum() or c in '_-' for c in run) and len(run)<=80
out=root/'artifacts'/'split'/run;out.mkdir(parents=True,exist_ok=False)
def remote(script):
    encoded=base64.b64encode(script.encode('utf-16le')).decode()
    return ['ssh','-o','BatchMode=yes',a.ssh,'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand '+encoded]
sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit local source before joint validation'
check=subprocess.check_output(remote(r'''$ProgressPreference='SilentlyContinue';Set-Location "$env:USERPROFILE\repos\Infer.NET";git rev-parse HEAD;git status --porcelain'''),text=True).strip()
assert check==sha,'Remote source is dirty or does not match local SHA'
config=subprocess.check_output(['ssh','-G',a.ssh],text=True)
peer=next(x.split(' ',1)[1] for x in config.splitlines() if x.startswith('hostname '))
with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:s.connect((peer,22));local=s.getsockname()[0]
secret=secrets.token_hex(32);env=os.environ.copy()
for k in list(env):
    if k.startswith('DEEPINFER_'):del env[k]
env.update(DEEPINFER_TOKEN=secret,DEEPINFER_METAL_KV='bf16',DEEPINFER_METAL_PREFILL_CHUNK='32',DEEPINFER_METAL_PREFILL_DELTA='sequential',DEEPINFER_METAL_PREFILL_GEMM='fp32')
# The verification configuration intentionally matches the serial CUDA precision path.
command=['dotnet',str(root/'src/DeepInfer.Host/bin/Release/net10.0/deepinfer.dll'),'split-infer','--allow-remote','--listen',f'tcp://{local}:{a.port}','--output',str(out),'--max-tokens',str(a.max_tokens),'--prompt',a.prompt]
if a.verify:command.append('--verify')
(out/'run.json').write_text(json.dumps(dict(source=sha,run=run,verify=a.verify,promptSha256=hashlib.sha256(a.prompt.encode()).hexdigest(),maxTokens=a.max_tokens,topology='Mac Wi-Fi -> home router -> Windows Ethernet',precision='affine8 / BF16 KV / FP32 recurrent and GEMV',startedUtc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
with (out/'mac.stdout.log').open('w') as stdout,(out/'mac.stderr.log').open('w') as stderr:
    mac=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr)
    (out/'mac.pid').write_text(str(mac.pid));deadline=time.monotonic()+600
    while 'SPLIT_LISTENING' not in (out/'mac.stderr.log').read_text():
        if mac.poll() is not None:raise RuntimeError('Mac startup/reference failed; see '+str(out))
        if time.monotonic()>deadline:mac.terminate();mac.wait();raise TimeoutError('Mac did not become ready')
        time.sleep(1)
    with (out/'ssh.log').open('w') as log:
        windows=subprocess.Popen(remote(r'''& "$env:USERPROFILE\repos\Infer.NET\scripts\windows\run-split-worker.ps1"'''),stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
        windows.stdin.write(json.dumps(dict(endpoint=f'tcp://{local}:{a.port}',token=secret,runId=run))+'\n');windows.stdin.close()
        print('Running',run,'source',sha,flush=True)
        deadline=time.monotonic()+1800
        try:
            while mac.poll() is None:
                if windows.poll() not in (None,0):raise RuntimeError('Windows worker failed; inspect remote artifacts/split/'+run)
                if time.monotonic()>deadline:raise TimeoutError('Split run deadline exceeded')
                time.sleep(1)
            remote_exit=windows.wait(timeout=60)
        finally:
            if mac.poll() is None:mac.terminate();mac.wait(timeout=30)
        (out/'exit.json').write_text(json.dumps(dict(mac=mac.returncode,windows=remote_exit)))
        print('Mac',mac.returncode,'Windows',remote_exit,'results',out,flush=True)
        raise SystemExit(mac.returncode or remote_exit)
