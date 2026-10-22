#!/usr/bin/env python3
"""Serial same-input attention query-batch A/B; never replaces a running service."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--batches', type=int, nargs='+', default=[128,256,512,1024,2048], choices=[128,256,512,1024,2048])
p.add_argument('--lengths', type=int, nargs='+', default=[8192,32768], choices=[8192,32768])
p.add_argument('--trials', type=int, default=1)
p.add_argument('--warmup', type=int, default=1)
p.add_argument('--tag', default='scout')
p.add_argument('--require-ac', action='store_true')
a = p.parse_args()
assert a.trials > 0 and a.warmup > 0 and a.tag.replace('-','').isalnum()
out = ROOT/'artifacts/baselines'
for length in a.lengths:
    material = ('qwen36-context-fixed-data/8192-4096-prompt.txt' if length == 8192 else
                'qwen36-context-large-fixed-data/32768-8192-prompt.txt')
    for batch in a.batches:
        prefix = f'qwen36-ab{batch}-{length//1024}k-{a.tag}'
        print(f'Starting {prefix}: {a.warmup} excluded full warmups, {a.trials} measured requests', flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/run-qwen36-context-suite.py'),
            '--gemm','bf16-nax','--gemm-impl','hybrid','--expert-impl','packed',
            '--attention-impl','bf16-nax','--kv','bf16','--key-tokens','128',
            '--attention-batch',str(batch),'--warmup',str(a.warmup),'--trials',str(a.trials),
            '--cases',f'{length}:128','--fixed-output','--keep-awake',
            '--prompt-file',str(out/material),'--prefix',prefix]+(['--require-ac'] if a.require_ac else []), cwd=ROOT, check=True)
        report = json.loads((out/(prefix+'.json')).read_text())
        case, = report['cases']
        print(json.dumps(dict(report=prefix,batch=batch,inputTokens=length,
            prefillMs=[r['timings']['prefillMs'] for r in case['trials']],
            allocatedBytes=case['trials'][0]['timings']['metalAllocatedBytes'])), flush=True)
