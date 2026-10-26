#!/usr/bin/env python3
"""Reconcile #92 fixed-request evidence without exposing prompts/private config."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(raw, output):
    manifest = read(raw / 'runtime-manifest.json')
    cohorts = []
    for shape, baseline in [('short', 'main-pre'), ('8192', 'main-pre'), ('32768', 'main-post')]:
        for arm in (baseline, 'candidate'):
            root = raw / f'{arm}-{shape}'
            rows = read(root / 'trials.json')
            workload = read(root / 'manifest.json')
            assert len(rows) == 4 and [r['trial'] for r in rows] == list(range(4))
            assert [r['warmup'] for r in rows] == [True, False, False, False]
            assert len({r['tokenHash'] for r in rows}) == 1
            assert read(raw / f'{arm}-{shape}.status.json')['exitCode'] == 0
            public = []
            for r in rows:
                response = read(root / f"response-{r['trial']}.json")
                t = r['timings']
                assert response['executionPath'] == 'metal-local'
                assert response['promptTokens'] == workload['inputTokens']
                assert len(response['tokenIds']) == workload['outputTokens'] == 128
                assert t['decodeTokens'] == 127 and t['decodeMs'] > 0 and t['prefillMs'] > 0
                assert response['timings'] == t and not r['profiles']
                digest = hashlib.sha256(json.dumps(response['tokenIds'], separators=(',', ':')).encode()).hexdigest()
                assert digest == r['tokenHash']
                public.append({k: r[k] for k in ['trial', 'warmup', 'clientSeconds', 'tokenHash',
                                                'swapoutDeltaBytes', 'metalAllocationBytes', 'resourcePass']} |
                              {'timings': {k: v for k, v in t.items() if k != 'decodeWindows'},
                               'outputTokenIds': response['tokenIds'],
                               'executionPath': response['executionPath']})
            formal = public[1:]
            prefill = sum(workload['inputTokens'] for _ in formal) / sum(r['timings']['prefillMs'] / 1000 for r in formal)
            decode = sum(r['timings']['decodeTokens'] for r in formal) / sum(r['timings']['decodeMs'] / 1000 for r in formal)
            cohorts.append({'shape': shape, 'arm': arm, 'sourceSha': manifest[arm]['sourceSha'],
                            'inputTokens': workload['inputTokens'], 'outputTokens': 128,
                            'inputSha256': workload['inputSha256'], 'driverSha256': workload['driverSha256'],
                            'rows': public, 'aggregatePrefillTokensPerSecond': prefill,
                            'aggregateDecodeTokensPerSecond': decode,
                            'medianPrefillSeconds': statistics.median(r['timings']['prefillMs'] / 1000 for r in formal),
                            'medianDecodeTokensPerSecond': statistics.median(r['timings']['decodeTokensPerSecond'] for r in formal),
                            'resourcePass': all(r['resourcePass'] for r in rows)})
    comparisons = []
    for shape in ['short', '8192', '32768']:
        a, b = [x for x in cohorts if x['shape'] == shape]
        assert a['inputSha256'] == b['inputSha256']
        comparisons.append({'shape': shape,
                            'mainPrefillTokensPerSecond': a['aggregatePrefillTokensPerSecond'],
                            'candidatePrefillTokensPerSecond': b['aggregatePrefillTokensPerSecond'],
                            'mainDecodeTokensPerSecond': a['aggregateDecodeTokensPerSecond'],
                            'candidateDecodeTokensPerSecond': b['aggregateDecodeTokensPerSecond'],
                            'decodeChangePercent': 100 * (b['aggregateDecodeTokensPerSecond'] / a['aggregateDecodeTokensPerSecond'] - 1),
                            'prefillLatencyChangePercent': 100 * (b['medianPrefillSeconds'] / a['medianPrefillSeconds'] - 1),
                            'identicalOutputTokenIds': a['rows'][0]['outputTokenIds'] == b['rows'][0]['outputTokenIds']})
    output.mkdir(parents=True, exist_ok=True)
    value = {'configuration': 'Mac-only affine8/group64, FP32 KV/recurrence, fused-fp32 decode, managed prefill32/scratch128; counters off, no prefix cache/speculation',
             'timing': 'Worker timings; sum input tokens / sum prefill seconds; sum N-1 intervals / sum decode seconds. Client wall time reported separately.',
             'order': 'main short/8K, candidate short/8K/32K, main 32K; one excluded warmup and three formal trials per cell',
             'cohorts': cohorts, 'comparisons': comparisons}
    (output / 'fixed.json').write_text(json.dumps(value, indent=2) + '\n')
    safe_runtime = {}
    for arm, data in manifest.items():
        safe_runtime[arm] = {k: data[k] for k in ['sourceSha', 'expertLanes', 'executableSha256', 'nativeSha256', 'driverSha256', 'version']}
        h = data['health']
        safe_runtime[arm]['health'] = {k: h[k] for k in ['state', 'ready', 'pid', 'build', 'revision', 'weightBits', 'kv'] if k in h}
        assert safe_runtime[arm]['sourceSha'] in h['build']
    (output / 'runtime-fixed.json').write_text(json.dumps(safe_runtime, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.raw, args.output)


if __name__ == '__main__':
    main()
