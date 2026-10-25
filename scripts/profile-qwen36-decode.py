#!/usr/bin/env python3
"""Collect fixed-request Metal decode diagnostics from an already isolated worker.

Raw responses are local evidence. This driver does not start/stop services or
change precision. Counter regions serialize work: stage shares describe the
instrumented run, not unperturbed hardware utilization or isolated barrier cost.
"""
import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path


def vm():
    raw = subprocess.check_output(['vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+)', raw)[1])
    counts = {k: int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)\.', raw, re.M)}
    return {'swapoutBytes': counts.get('Swapouts', 0) * page, 'raw': raw}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', default='http://127.0.0.1:5291')
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--expected-input', type=int, required=True)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--serve-log', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args()
    from urllib.parse import urlsplit
    u = urlsplit(a.endpoint)
    if u.scheme != 'http' or u.hostname not in ('127.0.0.1', 'localhost', '::1') or u.username or u.password or u.path or u.query or u.fragment:
        p.error('Require a credential-free loopback HTTP origin')
    if a.expected_input < 1 or a.max_tokens < 2:
        p.error('Require positive input length and at least two outputs')
    a.output.mkdir(parents=True, exist_ok=False)
    token = a.token_file.read_text().strip()
    if not token:
        p.error('Empty token')
    headers = {'X-DeepInfer-Token': token, 'Content-Type': 'application/json'}

    def request(path, body=None):
        req = urllib.request.Request(a.endpoint + path, headers=headers,
                                     data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)

    health = request('/health')
    if not health['ready'] or health['state'] != 'Ready' or health['kv'] != 'fp32':
        raise RuntimeError('Expected an idle ready FP32 worker')
    manifest = {'label': a.label, 'health': health, 'inputTokens': a.expected_input,
                'outputTokens': a.max_tokens, 'inputSha256': hashlib.sha256(a.prompt_file.read_bytes()).hexdigest(),
                'driverSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'windowsParticipates': False, 'precision': 'affine8 weights / FP32 KV and recurrence',
                'cache': 'none', 'speculation': 'off', 'warmups': 1, 'formalTrials': 3}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    for trial in range(4):
        power = subprocess.check_output(['pmset', '-g', 'batt'], text=True)
        if 'AC Power' not in power:
            raise RuntimeError('AC power required')
        before = vm()
        offset = a.serve_log.stat().st_size
        started = time.monotonic()
        result = request('/v1/infer', {'prompt': a.prompt_file.read_text(), 'maxTokens': a.max_tokens,
                         'model': 'qwen3.6-35b-a3b', 'backend': 'metal',
                         'benchmarkIgnoreEos': True, 'profile': True})
        elapsed = time.monotonic() - started
        after = vm()
        with a.serve_log.open('rb') as f:
            f.seek(offset)
            log = f.read().decode(errors='replace')
        (a.output / f'response-{trial}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        (a.output / f'worker-{trial}.log').write_text(log)
        if result['promptTokens'] != a.expected_input or len(result['tokenIds']) != a.max_tokens or result['executionPath'] != 'metal-local':
            raise RuntimeError('Input/output/topology mismatch; raw evidence retained')
        profiles = [json.loads(line.split('DECODE_STAGE_PROFILE ', 1)[1]) for line in log.splitlines() if 'DECODE_STAGE_PROFILE ' in line]
        swap = after['swapoutBytes'] - before['swapoutBytes']
        memory = result['timings']['metalAllocatedBytes']
        row = {'trial': trial, 'warmup': trial == 0, 'timings': result['timings'],
               'clientSeconds': elapsed, 'tokenHash': hashlib.sha256(json.dumps(result['tokenIds'], separators=(',', ':')).encode()).hexdigest(),
               'swapoutDeltaBytes': swap, 'beforeVm': before, 'afterVm': after, 'power': power,
               'profiles': profiles, 'metalAllocationBytes': memory,
               'resourcePass': memory <= 38.5 * 2**30 and (trial == 0 or swap <= 512 * 2**20)}
        rows.append(row)
        (a.output / 'trials.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps({k: row[k] for k in ('trial', 'warmup', 'clientSeconds', 'tokenHash', 'swapoutDeltaBytes', 'resourcePass')}) , flush=True)
        if not row['resourcePass']:
            raise RuntimeError('Resource gate failed; stop this cohort and retain evidence')
    if len({r['tokenHash'] for r in rows}) != 1:
        raise RuntimeError('Repeated token IDs differ')


if __name__ == '__main__':
    main()
