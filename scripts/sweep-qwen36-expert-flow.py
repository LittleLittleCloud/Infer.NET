#!/usr/bin/env python3
"""Same-binary expert-flow A/B, exact prompts, serial GPU work and power auditing."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--lengths', type=int, nargs='+', choices=[8192,32768], default=[8192,32768])
p.add_argument('--warmup', type=int, default=3)
p.add_argument('--trials', type=int, default=3)
p.add_argument('--tag', default='confirm')
p.add_argument('--pipeline-experiment', action='store_true')
a = p.parse_args()
assert a.warmup > 0 and a.trials > 0 and a.tag.replace('-','').isalnum()
source = subprocess.check_output(['pmset','-g','batt'], text=True).splitlines()[0]
settings = subprocess.check_output(['pmset','-g','custom'], text=True).strip()
out = ROOT/'artifacts/baselines'
for length in a.lengths:
    configs = [('control','staged','serial'),('direct','direct','serial')]
    if length == 32768: configs.reverse()
    if a.pipeline_experiment: configs.append(('pipeline','direct','double-buffer'))
    material = ('qwen36-context-fixed-data/8192-4096-prompt.txt' if length==8192 else
                'qwen36-context-large-fixed-data/32768-8192-prompt.txt')
    for label,flow,pipeline in configs:
        prefix=f'qwen36-expert-flow-{label}-{length//1024}k-{a.tag}'
        print('Starting',prefix,source,flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/run-qwen36-context-suite.py'),
            '--gemm','bf16-nax','--gemm-impl','hybrid','--expert-impl','packed',
            '--expert-flow',flow,'--expert-pipeline',pipeline,
            '--attention-impl','bf16-nax','--kv','bf16','--key-tokens','128',
            '--warmup',str(a.warmup),'--trials',str(a.trials),'--cases',f'{length}:128',
            '--fixed-output','--keep-awake','--prompt-file',str(out/material),'--prefix',prefix],cwd=ROOT,check=True)
        report=json.loads((out/(prefix+'.json')).read_text());case,=report['cases']
        observations=list(case['memorySamples'])
        for sample in case['warmups']+case['trials']: observations += [sample['powerBefore'],sample['powerAfter']]
        assert all((s.get('power') or '').splitlines()[0] == source and s.get('powerSettings') == settings for s in observations), 'Power source/settings changed; do not combine this round'
        print(json.dumps(dict(report=prefix,prefillMs=[s['timings']['prefillMs'] for s in case['trials']],powerObservations=len(observations))),flush=True)
