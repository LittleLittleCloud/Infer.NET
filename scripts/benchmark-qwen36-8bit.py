#!/usr/bin/env python3
"""Serial original-engine 8-bit baseline. Run numerical verification first; no external inference runtime."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--host', type=Path, default=ROOT/'artifacts/native/qwen36-8bit/deepinfer')
p.add_argument('--model', type=Path, default=ROOT/'models/qwen3.6-35b-a3b-8bit-mlx')
p.add_argument('--trials', type=int, default=3)
p.add_argument('--long-output', type=int, default=4096)
p.add_argument('--chunk', type=int, choices=[32,64,128,256,512,1024,2048], default=2048)
p.add_argument('--long-chunk', type=int, choices=[32,64,128,256,512,1024,2048], default=2048,
               help='128K prefill chunk; smaller values reduce scratch without changing precision')
p.add_argument('--long-output-chunk', type=int, choices=[32,64,128,256,512,1024,2048], default=512,
               help='128K + long-output needs additional KV space; use a smaller workspace')
p.add_argument('--warmup', type=int, default=1)
p.add_argument('--skip-long-output', action='store_true')
p.add_argument('--label', default='', help='Optional unique run label; preserves older binary results')
p.add_argument('--lengths', nargs='+', type=int, choices=[8192,32768,131072], default=[8192,32768,131072])
a = p.parse_args()
if a.trials < 1 or a.warmup < 0 or not 129 <= a.long_output <= 32768:
    p.error('trials >= 1, warmup >= 0, long-output 129..32768 required')
if a.label and not re.fullmatch('[a-z0-9-]+', a.label):
    p.error('label must contain only lowercase letters, numbers and hyphens')
config = json.loads((a.model/'config.json').read_text())
assert config['quantization']['bits'] == 8
assert json.loads((ROOT/'artifacts/reference/qwen36-8bit/comparison.json').read_text())['passed']
reference = json.loads((ROOT/'artifacts/reference/qwen36-8bit/reference.json').read_text())
assert reference['provenance'] == json.loads((a.model/'provenance.json').read_text())
out = ROOT/'artifacts/baselines'
prompts = {
    8192: out/'qwen36-context-fixed-data/8192-4096-prompt.txt',
    32768: out/'qwen36-context-large-fixed-data/32768-8192-prompt.txt',
    131072: out/'qwen36-context-plan-data/131072-8192-prompt.txt',
}
jobs = [(n,128,a.warmup,a.trials) for n in a.lengths]
if 131072 in a.lengths and not a.skip_long_output:
    jobs.append((131072,a.long_output,a.warmup,a.trials))
for n,output,warmup,trials in jobs:
    chunk = a.long_chunk if n >= 131072 else a.chunk
    if output > 128:
        chunk = a.long_output_chunk
    chunk_label = f'-chunk{chunk}' if chunk != 2048 else ''
    label = a.label+'-' if a.label else ''
    name = f'qwen36-8bit-{label}{n//1024}k{chunk_label}-{output}out'
    if (out/(name+'.json')).exists():
        raise FileExistsError(f'Preserve or archive existing results before rerunning: {name}.json')
    acceptance = ['--acceptance-reference',str(ROOT/'artifacts/reference/qwen36-8bit/reference.json')] if n == 131072 and output == a.long_output else []
    subprocess.run([sys.executable,str(ROOT/'scripts/run-qwen36-context-suite.py'),
        '--host',str(a.host.resolve()),'--model-directory',str(a.model.resolve()),'--base-port','5400',
        '--gemm','bf16-nax','--gemm-impl','hybrid','--expert-impl','packed',
        '--expert-flow','staged','--expert-pipeline','serial','--expert-tiles','adaptive',
        '--expert-gate','fused','--delta-impl','device','--attention-impl','bf16-nax',
        '--kv','bf16','--key-tokens','128','--attention-partition','1024',
        '--chunk',str(chunk),
        '--warmup',str(warmup),'--trials',str(trials),'--cases',f'{n}:{output}',
        '--fixed-output','--keep-awake','--require-ac','--prompt-file',str(prompts[n]),
        '--prefix',name,*acceptance],cwd=ROOT,check=True)
    print('Completed',name,flush=True)
