#!/usr/bin/env python3
"""Serial, same-binary A/B for the selected M5 prefill changes."""
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'artifacts/baselines'
power = subprocess.check_output(['pmset','-g','batt'],text=True).splitlines()[0]
settings = subprocess.check_output(['pmset','-g','custom'],text=True).strip()
assert "'AC Power'" in power, 'This comparison requires AC power'
for length in [8192,32768]:
    configs = [('control','fixed','split','staged'),('selected','adaptive','fused','device')]
    if length==32768: configs.reverse()
    material = ('qwen36-context-fixed-data/8192-4096-prompt.txt' if length==8192 else
                'qwen36-context-large-fixed-data/32768-8192-prompt.txt')
    for label,tiles,gate,delta in configs:
        prefix=f'qwen36-joy-{label}-{length//1024}k'
        print('Starting',prefix,flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/run-qwen36-context-suite.py'),
            '--gemm','bf16-nax','--gemm-impl','hybrid','--expert-impl','packed',
            '--expert-flow','staged','--expert-pipeline','serial',
            '--expert-tiles',tiles,'--expert-gate',gate,'--delta-impl',delta,
            '--attention-impl','bf16-nax','--kv','bf16','--key-tokens','128','--attention-partition','1024',
            '--warmup','3','--trials','3','--cases',f'{length}:128','--fixed-output','--keep-awake','--require-ac',
            '--prompt-file',str(out/material),'--prefix',prefix],cwd=ROOT,check=True)
        report=json.loads((out/(prefix+'.json')).read_text());case,=report['cases']
        observations=list(case['memorySamples'])
        for sample in case['warmups']+case['trials']: observations += [sample['powerBefore'],sample['powerAfter']]
        assert all((s.get('power') or '').splitlines()[0]==power and s.get('powerSettings')==settings for s in observations), 'Power changed during A/B'
        print(prefix,[s['timings']['prefillMs'] for s in case['trials']],flush=True)
