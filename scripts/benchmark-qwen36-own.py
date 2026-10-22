#!/usr/bin/env python3
"""Measure the running Native AOT coordinator/worker using stdlib HTTP only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--endpoint', default='http://127.0.0.1:5100')
parser.add_argument('--trials', type=int, default=3)
parser.add_argument('--max-tokens', type=int, default=128)
parser.add_argument('--expected-submissions-per-forward', type=int, choices=[1, 191, 231, 541], default=1,
                    help='1 for GPU graph; 191 for host executor; 231 for EXPERT_UNFUSED=1; 541 for METAL_SERIAL=1')
parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/baselines/qwen36-gpu-graph.json')
parser.add_argument('--prefill-mode',choices=['batched','sequential'],default='batched')
parser.add_argument('--prefill-chunk',type=int,choices=[32,64,128,256,512,1024,2048],default=2048)
args = parser.parse_args()
if args.trials < 1 or not 2 <= args.max_tokens <= 256:
    parser.error('Use trials >= 1 and max-tokens between 2 and 256')
payload = {'prompt': '请用中文详细解释什么是机器学习，并举出三个实际应用。',
           'maxTokens': args.max_tokens, 'model': 'qwen3.6-35b-a3b', 'backend': 'metal'}
headers = {'Content-Type': 'application/json'}
if os.environ.get('DEEPINFER_TOKEN'):
    headers['X-DeepInfer-Token'] = os.environ['DEEPINFER_TOKEN']

def infer():
    start = time.perf_counter()
    request = urllib.request.Request(args.endpoint+'/v1/infer', json.dumps(payload).encode(), headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    if result['isSimulation'] or not result['gpuDispatches'] or result['model'] != payload['model']:
        raise ValueError('Expected real own-engine MoE inference')
    if len(result['tokenIds']) != args.max_tokens or result['finishReason'] != 'length':
        raise ValueError('This fixed-length benchmark requires all requested output tokens')
    forwards = result['promptTokens'] + len(result['tokenIds']) - 1
    # Prompt tokens except the last omit the classifier; generation needs one per output.
    dispatches_per_forward = {1:675,191:501}.get(args.expected_submissions_per_forward,541)
    if args.expected_submissions_per_forward==1 and args.prefill_mode=='batched':
        chunks=(result['promptTokens']+args.prefill_chunk-1)//args.prefill_chunk
        expected_dispatches=chunks*672+3+(len(result['tokenIds'])-1)*675
        expected_submissions=chunks+len(result['tokenIds'])-1
    else:
        expected_dispatches = forwards * (672 if args.expected_submissions_per_forward==1 else dispatches_per_forward - 1) + len(result['tokenIds']) * (3 if args.expected_submissions_per_forward==1 else 1)
        expected_submissions = forwards if args.expected_submissions_per_forward==1 else forwards * (args.expected_submissions_per_forward - 1) + len(result['tokenIds'])
    if result['gpuDispatches'] != expected_dispatches or result.get('gpuSubmissions') != expected_submissions:
        raise ValueError(f'Unexpected kernel/submission counts: {result["gpuDispatches"]}/{result.get("gpuSubmissions")}')
    return {'rpcMs': (time.perf_counter()-start)*1000, 'result': result}

warmup = infer()
trials = []
for n in range(args.trials):
    trial = infer()
    if trial['result']['tokenIds'] != warmup['result']['tokenIds']:
        raise ValueError('Fresh-state deterministic generation mismatch')
    trials.append(trial)
    print(f"Trial {n+1}: {trial['result']['timings']}", flush=True)
timings = [t['result']['timings'] for t in trials]
report = {'completedAtUtc': datetime.now(timezone.utc).isoformat(),
          'engine': 'DeepInfer.NET Native AOT + original Metal GPU graph' if args.expected_submissions_per_forward==1 else 'DeepInfer.NET Native AOT + original Metal kernels, host executor',
          'macOS': platform.mac_ver()[0], 'machine': platform.machine(),
          'precision': '5-bit/group64 affine, 8-bit routers, FP32 activations/state/accumulation',
          'submissionsPerDecodeForward': args.expected_submissions_per_forward,
          'dispatchesPerDecodeForward': {1:675,191:501}.get(args.expected_submissions_per_forward,541),
          'prefillMode':args.prefill_mode if args.expected_submissions_per_forward==1 else 'host-sequential',
          'prefillChunkTokens':args.prefill_chunk,
          'provenanceSha256': hashlib.sha256((ROOT/'models/qwen3.6-35b-a3b-5bit-mlx/provenance.json').read_bytes()).hexdigest(),
          'request': payload, 'warmup': warmup, 'trials': trials,
          'decodeTokensPerSecond': sum(t['decodeTokens'] for t in timings)*1000/sum(t['decodeMs'] for t in timings),
          'meanPrefillMs': sum(t['prefillMs'] for t in timings)/len(timings),
          'meanFirstTokenMs': sum(t['firstTokenMs'] for t in timings)/len(timings),
          'meanRpcMs': sum(t['rpcMs'] for t in trials)/len(trials)}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('warmup','trials')}, ensure_ascii=False, indent=2))
