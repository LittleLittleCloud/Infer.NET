#!/usr/bin/env python3
"""Own a single model cluster for a long benchmark; optionally restore an idle live cluster after completion/failure."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-directory', type=Path, default=ROOT/'models/qwen3.6-35b-a3b-5bit-mlx')
    p.add_argument('--acceptance-reference', type=Path, help='Run real RPC reset/needle checks after benchmarking, before stopping this worker')
    p.add_argument('--cases', nargs='+', default=['270000:8192', '245760:16384'])
    p.add_argument('--base-port', type=int, default=5100)
    p.add_argument('--fixed-output', action='store_true')
    p.add_argument('--restore-live', action='store_true')
    p.add_argument('--keep-awake', action='store_true', help='Hold an idle-sleep assertion only while the managed job runs')
    p.add_argument('--require-ac', action='store_true')
    p.add_argument('--prefix', default='qwen36-context-full')
    p.add_argument('--corpus', type=Path)
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--attention', choices=['legacy','fused'], default='fused')
    p.add_argument('--experts', choices=['vector','grouped'], default='grouped')
    p.add_argument('--gemm', choices=['fp32','bf16-nax'], default=os.environ.get('DEEPINFER_METAL_PREFILL_GEMM','fp32'))
    p.add_argument('--gemm-impl', choices=['staged','hybrid','expanded'], default=os.environ.get('DEEPINFER_METAL_PREFILL_GEMM_IMPL','staged'))
    p.add_argument('--expert-impl',choices=['staged','packed'],default=os.environ.get('DEEPINFER_METAL_PREFILL_EXPERT_IMPL','staged'))
    p.add_argument('--expert-flow',choices=['staged','direct'],default=os.environ.get('DEEPINFER_METAL_PREFILL_EXPERT_FLOW','staged'))
    p.add_argument('--expert-pipeline',choices=['serial','double-buffer'],default=os.environ.get('DEEPINFER_METAL_PREFILL_EXPERT_PIPELINE','serial'))
    p.add_argument('--attention-impl', choices=['fp32','bf16-nax'], default=os.environ.get('DEEPINFER_METAL_PREFILL_ATTENTION_IMPL','fp32'))
    p.add_argument('--kv', choices=['fp32','bf16'], default=os.environ.get('DEEPINFER_METAL_KV','fp32'))
    p.add_argument('--delta', choices=['sequential','chunked'], default='chunked')
    p.add_argument('--chunk', type=int, choices=[32,64,128,256,512,1024,2048], default=2048)
    p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--key-tokens',type=int,choices=[32,64,128],default=int(os.environ.get('DEEPINFER_METAL_PREFILL_KEY_TILE','32')))
    p.add_argument('--attention-batch',type=int,choices=[128,256,512,1024,2048],default=int(os.environ.get('DEEPINFER_METAL_PREFILL_ATTENTION_BATCH','128')))
    p.add_argument('--delta-impl',choices=['staged','device'],default=os.environ.get('DEEPINFER_METAL_PREFILL_DELTA_IMPL','staged'))
    p.add_argument('--expert-gate',choices=['split','fused'],default=os.environ.get('DEEPINFER_METAL_PREFILL_EXPERT_GATE','split'))
    p.add_argument('--expert-tiles',choices=['fixed','adaptive'],default=os.environ.get('DEEPINFER_METAL_PREFILL_EXPERT_TILES','fixed'))
    p.add_argument('--attention-partition',type=int,choices=[1024,2048,4096,8192,16384,32768],default=int(os.environ.get('DEEPINFER_METAL_PREFILL_ATTENTION_PARTITION','1024')))
    p.add_argument('--query-tokens',type=int,choices=[4,8],default=4)
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--host', type=Path, default=ROOT/'artifacts/native/osx-arm64/deepinfer')
    a = p.parse_args()
    if not 1024 <= a.base_port <= 65000 or not re.fullmatch('[a-z0-9-]+', a.prefix):
        p.error('Invalid port/prefix')
    # Never stop or replace a pre-existing user service.
    for port in (a.base_port, a.base_port+1):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
    out = ROOT/'artifacts/baselines'; out.mkdir(parents=True, exist_ok=True)
    status_path = out/(a.prefix+'-job.json')
    status = dict(supervisorPid=os.getpid(), status='starting', cases=a.cases, gemm=a.gemm, kv=a.kv, gemmImpl=a.gemm_impl, attentionImpl=a.attention_impl, attentionBatch=a.attention_batch,
                  endpoint=f'http://127.0.0.1:{a.base_port}',
                  reportFile=str(out/(a.prefix+'.json')), startedAtUtc=datetime.now(timezone.utc).isoformat())
    def save():
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2)+'\n'); temp.replace(status_path)
    env={**os.environ, "DEEPINFER_METAL_PREFILL_ATTENTION": a.attention, "DEEPINFER_METAL_PREFILL_CHUNK": str(a.chunk), "DEEPINFER_METAL_PREFILL_EXPERTS": a.experts, "DEEPINFER_METAL_PREFILL_DELTA": a.delta, "DEEPINFER_METAL_PREFILL_GEMM": a.gemm, "DEEPINFER_METAL_KV": a.kv, "DEEPINFER_METAL_PREFILL_GEMM_IMPL": a.gemm_impl, "DEEPINFER_METAL_PREFILL_ATTENTION_IMPL": a.attention_impl,"DEEPINFER_METAL_PREFILL_QUERY_TILE":str(a.query_tokens),"DEEPINFER_METAL_PREFILL_EXPERT_IMPL":a.expert_impl,"DEEPINFER_METAL_PREFILL_KEY_TILE":str(a.key_tokens),"DEEPINFER_METAL_PREFILL_ATTENTION_BATCH":str(a.attention_batch)}
    def cluster():
        log = (out/(a.prefix+'-cluster.log')).open('a')
        env['DEEPINFER_METAL_PREFILL_DELTA_IMPL'] = a.delta_impl
        env['DEEPINFER_METAL_PREFILL_EXPERT_GATE'] = a.expert_gate
        env['DEEPINFER_METAL_PREFILL_EXPERT_TILES'] = a.expert_tiles
        if a.attention_partition != 1024: env['DEEPINFER_METAL_PREFILL_ATTENTION_PARTITION'] = str(a.attention_partition)
        else: env.pop('DEEPINFER_METAL_PREFILL_ATTENTION_PARTITION', None)
        env['DEEPINFER_METAL_PREFILL_EXPERT_FLOW'] = a.expert_flow
        env['DEEPINFER_METAL_PREFILL_EXPERT_PIPELINE'] = a.expert_pipeline
        try:
            return subprocess.Popen([str(a.host.resolve()), 'demo',
                '--model', 'qwen3.6-35b-a3b', '--weights', str(a.model_directory.resolve()), '--base-port', str(a.base_port), '--stay'],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log.close()
    def stop(process):
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    service = benchmark = wake = None
    headers = {'X-DeepInfer-Token': os.environ['DEEPINFER_TOKEN']} if os.environ.get('DEEPINFER_TOKEN') else {}
    save()
    try:
        if a.keep_awake:
            wake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            status['idleSleepAssertionPid'] = wake.pid
        service = cluster(); status['clusterPid'] = service.pid; save()
        if a.base_port == 5100:
            for name in ['prefill-live-cluster.pid', 'gpu-graph-live-cluster.pid', 'long-context-live-cluster.pid']:
                (ROOT/'artifacts'/name).write_text(str(service.pid)+'\n')
        for _ in range(600):
            if service.poll() is not None:
                raise RuntimeError('Cluster exited; inspect cluster log')
            try:
                with urllib.request.urlopen(urllib.request.Request(status['endpoint']+'/v1/workers', headers=headers), timeout=1) as response:
                    workers = json.load(response)['workers']
                if any(w['healthy'] and any(m['id'] == 'qwen3.6-35b-a3b' for m in w['registration']['models']) for w in workers):
                    break
            except OSError:
                pass
            time.sleep(.2)
        else:
            raise TimeoutError('Model startup timed out')
        command = [sys.executable, str(ROOT/'scripts/benchmark-qwen36-context.py'), '--run',
                   '--model-directory', str(a.model_directory.resolve()), '--endpoint', status['endpoint'], '--host', str(a.host.resolve()), '--attention', a.attention, '--chunk', str(a.chunk), '--experts', a.experts, '--delta', a.delta, '--gemm', a.gemm, '--kv', a.kv, '--gemm-impl', a.gemm_impl, '--attention-impl', a.attention_impl, '--key-tokens',str(a.key_tokens),'--expert-impl',a.expert_impl,'--query-tokens',str(a.query_tokens),'--attention-batch',str(a.attention_batch),'--warmup',str(a.warmup),'--trials', str(a.trials), '--cases', *a.cases, '--output', status['reportFile']]
        for line in subprocess.check_output(['ps', '-axo', 'pid,ppid,command'], text=True).splitlines()[1:]:
            fields = line.split(None, 2)
            if len(fields) == 3 and fields[1] == str(service.pid) and ' worker ' in fields[2]:
                status['workerPid'] = int(fields[0]); command += ['--worker-pid', fields[0]]
                break
        if a.fixed_output:
            command.append('--fixed-output')
        command += ['--delta-impl', a.delta_impl]
        command += ['--expert-gate', a.expert_gate]
        command += ['--expert-tiles', a.expert_tiles, '--attention-partition', str(a.attention_partition)]
        command += ['--expert-flow', a.expert_flow]
        command += ['--expert-pipeline', a.expert_pipeline]
        if a.require_ac:
            command.append('--require-ac')
        if a.prompt_file:
            command += ["--prompt-file", str(a.prompt_file.resolve())]
        if a.corpus:
            command += ['--corpus', str(a.corpus.resolve())]
        with (out/(a.prefix+'-benchmark.log')).open('w') as log:
            benchmark = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status.update(status='running', benchmarkPid=benchmark.pid); save()
            code = benchmark.wait()
        if code == 0 and a.acceptance_reference:
            subprocess.run([sys.executable, str(ROOT/'scripts/accept-qwen36-long.py'),
                '--endpoint', status['endpoint'], '--reference', str(a.acceptance_reference.resolve()),
                '--output', str(out/(a.prefix+'-acceptance.json'))], cwd=ROOT, env=env, check=True)
        status.update(status='completed' if code == 0 else 'failed', benchmarkExitCode=code)
    except KeyboardInterrupt:
        status.update(status='cancelled')
    except Exception as error:
        status.update(status='failed', error=repr(error))
    finally:
        stop(benchmark); stop(service)
        status['finishedAtUtc'] = datetime.now(timezone.utc).isoformat(); save()
        if a.restore_live:
            try:
                live = cluster(); status['restoredLiveClusterPid'] = live.pid
                status['restoredLiveHealthy'] = False; save()
                if a.base_port == 5100:
                    for name in ['prefill-live-cluster.pid', 'gpu-graph-live-cluster.pid', 'long-context-live-cluster.pid']:
                        (ROOT/'artifacts'/name).write_text(str(live.pid)+'\n')
                for _ in range(600):
                    if live.poll() is not None:
                        status['restoreError'] = 'Restored cluster exited; inspect cluster log'; break
                    try:
                        request = urllib.request.Request(status['endpoint']+'/v1/workers', headers=headers)
                        with urllib.request.urlopen(request, timeout=1) as response:
                            current = json.load(response)['workers']
                        if any(w['healthy'] and w['registration']['models'] for w in current):
                            status['restoredLiveHealthy'] = True; break
                    except OSError:
                        pass
                    time.sleep(.2)
                if not status['restoredLiveHealthy']:
                    status.setdefault('restoreError', 'Restored model startup timed out')
            except Exception as error:
                status['restoreError'] = repr(error)
        save()
        subprocess.run([sys.executable, str(ROOT/'scripts/summarize-qwen36-context.py')], cwd=ROOT, check=False)
        stop(wake)
    return 0 if status['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
