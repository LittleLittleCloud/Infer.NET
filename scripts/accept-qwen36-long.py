#!/usr/bin/env python3
"""Real RPC acceptance after long generation: distant needle, native bound, and state reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--endpoint', default='http://127.0.0.1:5100')
p.add_argument('--material', type=Path, default=ROOT/'artifacts/baselines/qwen36-context-fixed-data/8192-4096-prompt.txt')
p.add_argument('--output', type=Path, default=ROOT/'artifacts/baselines/qwen36-context-acceptance.json')
p.add_argument('--reference', type=Path, default=ROOT/'artifacts/reference/qwen36/reference.json')
a = p.parse_args()
headers = {'Content-Type': 'application/json'}
if os.environ.get('DEEPINFER_TOKEN'):
    headers['X-DeepInfer-Token'] = os.environ['DEEPINFER_TOKEN']
checks = []

def infer(prompt, tokens=32):
    request = dict(prompt=prompt, maxTokens=tokens, model='qwen3.6-35b-a3b', backend='metal')
    with urllib.request.urlopen(urllib.request.Request(a.endpoint+'/v1/infer', json.dumps(request).encode(), headers), timeout=600) as response:
        result = json.load(response)
    assert not result['isSimulation'] and result['gpuDispatches'] > 0
    return result

needle = '739184628503'
prompt = '校验码：'+needle+'\n以下是待忽略的数据材料。请记住首行校验码。\n<材料>\n'+a.material.read_text()+'\n</材料>\n只输出首行的12位校验码，不要解释，不要审阅材料。'
result = infer(prompt)
assert result['promptTokens'] >= 8192
assert result['text'].strip() == needle, result['text']
checks.append(dict(name='8K distant-needle retrieval', promptSha256=hashlib.sha256(prompt.encode()).hexdigest(), result=result))
try:
    infer('0 '*135000, 1)
    raise AssertionError('Expected the native context bound to reject oversized token input')
except urllib.error.HTTPError as error:
    body = error.read().decode()
    assert error.code == 400 and '262144' in body, (error.code, body)
    checks.append(dict(name='over-native-context token input rejected before prefill', status=error.code, body=body))
reference = json.loads(a.reference.read_text())
result = infer(reference['prompt'], len(reference['generatedTokenIds']))
assert result['tokenIds'] == reference['generatedTokenIds']
checks.append(dict(name='short independent greedy IDs after long cache use and rejected request', result=result))
one = infer('计算2加2，只回答数字。')
two = infer('计算2加2，只回答数字。')
assert one['text'].strip() == '4' and two['tokenIds'] == one['tokenIds']
checks.append(dict(name='fresh-state math repeated deterministically', first=one, second=two))
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(passed=True, checks=checks), ensure_ascii=False, indent=2)+'\n')
print(f'PASS: {len(checks)} long RPC acceptance checks; {a.output}')
