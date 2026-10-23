"""Local diagnostic for an already-running split deployment; no process changes.

The runner records each request, power and system swap counters. It is not the
formal 128K/4096 acceptance suite and does not infer CUDA-only time from TTFT.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['batch-fp32-packed', 'batch-fp32-score-reuse'], required=True)
p.add_argument('--prompt-file', required=True, type=pathlib.Path)
p.add_argument('--prompt-tokens', required=True, type=int)
p.add_argument('--token-file', required=True, type=pathlib.Path)
p.add_argument('--output', required=True, type=pathlib.Path)
p.add_argument('--rounds', type=int, default=4)
p.add_argument('--source-prefix', required=True)
a = p.parse_args()
assert a.rounds >= 2
a.output.mkdir(parents=True, exist_ok=False)
key = a.token_file.read_text().strip()
headers = {'Content-Type': 'application/json', 'X-DeepInfer-Token': key}

def get(path):
    request = urllib.request.Request('http://127.0.0.1:5200' + path, headers=headers)
    return json.load(urllib.request.urlopen(request, timeout=10))

def vm():
    output = subprocess.check_output(['vm_stat'], text=True)
    return {match.group(1).strip('"'): int(match.group(2))
            for match in re.finditer(r'^([^:]+):\s+(\d+)\.', output, re.M)}

def power():
    return subprocess.check_output(['pmset', '-g', 'batt'], text=True).strip()

health = json.load(urllib.request.urlopen('http://127.0.0.1:5201/health', timeout=10))
workers = get('/v1/workers')
assert health['ready'] and health['state'] == 'Ready' and health['weightBits'] == 8 and health['kv'] == 'fp32'
assert a.source_prefix in health['build'], health['build']
assert len(workers['workers']) == 1 and workers['workers'][0]['registration']['telemetry']['agentConnected']
assert 'AC Power' in power()
prompt_bytes = a.prompt_file.read_bytes()
request = {'prompt': prompt_bytes.decode(), 'maxTokens': 16, 'model': 'qwen3.6-35b-a3b',
           'backend': 'metal', 'profile': True, 'benchmarkIgnoreEos': True}
body = json.dumps(request).encode()
metadata = {'mode': a.mode, 'promptSha256': hashlib.sha256(prompt_bytes).hexdigest(),
            'promptTokens': a.prompt_tokens, 'maxTokens': 16, 'warmups': 1,
            'measuredRounds': a.rounds - 1, 'kind': 'same-package diagnostic; excludes GPU peak sampling'}
(a.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
(a.output / 'health.json').write_text(json.dumps(health, indent=2) + '\n')
(a.output / 'workers.json').write_text(json.dumps(workers, indent=2) + '\n')
records = []
for trial in range(a.rounds):
    before_power = power()
    assert 'AC Power' in before_power
    before_vm = vm()
    start = time.monotonic()
    req = urllib.request.Request('http://127.0.0.1:5200/v1/infer', data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as response:
        result = json.load(response)
    elapsed = time.monotonic() - start
    after_vm = vm()
    after_power = power()
    assert 'AC Power' in after_power
    assert result['promptTokens'] == a.prompt_tokens and len(result['tokenIds']) == 16
    assert result['executionPath'] == 'cuda-metal-split'
    if records:
        assert result['tokenIds'] == records[0]['tokenIds'], 'Token divergence within mode'
    record = {'trial': trial, 'warmup': trial == 0, 'clientSeconds': elapsed,
              'prefillMs': result['timings']['prefillMs'],
              'decodeTokensPerSecond': result['timings']['decodeTokensPerSecond'],
              'tokenIds': result['tokenIds'],
              'gpuComputeMs': result.get('gpuComputeMs'),
              'metalAllocatedBytes': result['timings']['metalAllocatedBytes'],
              'swapOutMiB': (after_vm['Swapouts'] - before_vm['Swapouts']) / 64,
              'powerBefore': before_power, 'powerAfter': after_power,
              'timings': result['timings']}
    records.append(record)
    (a.output / f'trial-{trial}.json').write_text(json.dumps(result, indent=2))
    (a.output / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('trial', 'warmup', 'clientSeconds', 'prefillMs', 'decodeTokensPerSecond', 'swapOutMiB')}), flush=True)
print('PASS', a.output, flush=True)
