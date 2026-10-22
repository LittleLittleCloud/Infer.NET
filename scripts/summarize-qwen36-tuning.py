#!/usr/bin/env python3
"""Audit the selected expert/attention tuning against its predecessor and MLX."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/baselines'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(stem):
    path = OUT / (stem + '.json')
    return json.loads(path.read_text()), dict(path=str(path), sha256=sha(path))


def own(stem, current=False):
    report, evidence = read(stem)
    case, = report['cases']
    assert case['status'] == 'completed'
    assert report['excludedWarmupRequests'] == 1
    assert len(case['warmups']) == 1 and len(case['trials']) == 3
    ids = []
    for sample in case['warmups'] + case['trials']:
        path = Path(sample['rawResultFile'])
        assert sha(path) == sample['rawResultSha256']
        raw = json.loads(path.read_text())
        assert not raw['isSimulation'] and len(raw['tokenIds']) == 128
        assert sample['timings']['decodeTokens'] == 127
        ids.append(raw['tokenIds'])
    if current:
        published = ROOT / 'artifacts/native/osx-arm64'
        assert sha(published / 'deepinfer') == report['nativeExecutableSha256']
        assert sha(published / 'libdeepinfer_metal.dylib') == report['nativeMetalLibrarySha256']
    samples = case['trials']
    return dict(evidence=evidence, inputTokens=case['inputTokens'], outputTokens=128,
        promptSha256=case['promptSha256'], promptTokenIdsSha256=case['promptTokenIdsSha256'],
        prefillMs=[s['timings']['prefillMs'] for s in samples],
        meanPrefillMs=statistics.mean(s['timings']['prefillMs'] for s in samples),
        meanFirstTokenMs=statistics.mean(s['timings']['firstTokenMs'] for s in samples),
        decodeTokensPerSecond=381000 / sum(s['timings']['decodeMs'] for s in samples),
        identicalMeasuredOutputIds=all(value == ids[1] for value in ids[1:]),
        precision=report['precision'], excludedWarmupMs=case['warmups'][0]['timings']['prefillMs'],
        executableSha256=report['nativeExecutableSha256'], librarySha256=report['nativeMetalLibrarySha256'])


def mlx(stem):
    report, evidence = read(stem)
    row, = report['rows']
    assert report['completedAtUtc'] and len(row['samples']) == 3 and len(row['warmups']) == 1
    assert report['requestedGenerationTokens'] == 128 and report['warmupPerCase'] == 1
    assert report['forceOutputLength'] and not report['prefixCache']
    assert all(entry['matchesInstalledRecord'] for entry in report['runtimeModuleHashes'].values())
    provenance = ROOT / 'models/qwen3.6-35b-a3b-5bit-mlx/provenance.json'
    assert report['provenanceSha256'] == sha(provenance)
    return dict(evidence=evidence, inputTokens=row['promptTokens'], outputTokens=128,
        promptSha256=row['promptSha256'], promptTokenIdsSha256=row['promptTokenIdsSha256'],
        meanPrefillMs=statistics.mean(s['mlxPromptMs'] for s in row['samples']),
        meanFirstTokenMs=statistics.mean(s['ttftMs'] for s in row['samples']),
        decodeTokensPerSecond=row['summary']['decodeTokensPerSecond'],
        versions=report['versions'], runtimeModuleHashes=report['runtimeModuleHashes'],
        runtimeEnvironment=report['runtimeEnvironment'], precision=report['floatPrecision'])


def main():
    rows = []
    for length, label in [(8192, '8k'), (32768, '32k')]:
        before = own('qwen36-key128-release' + label)
        selected = own('qwen36-tuned-release' + label, current=True)
        baseline = mlx('qwen36-tuned-mlx' + label)
        for field in ['inputTokens', 'outputTokens', 'promptSha256', 'promptTokenIdsSha256']:
            assert before[field] == selected[field] == baseline[field], field
        rows.append(dict(inputTokens=length, before=before, selected=selected, mlx=baseline,
            ownPromptTokensPerSecond=length * 1000 / selected['meanPrefillMs'],
            mlxPromptTokensPerSecond=length * 1000 / baseline['meanPrefillMs'],
            prefillTimeReduction=1 - selected['meanPrefillMs'] / before['meanPrefillMs'],
            prefillTimeRatioToMlx=selected['meanPrefillMs'] / baseline['meanPrefillMs']))
    acceptance, acceptance_evidence = read('qwen36-tuned-acceptance')
    assert acceptance['passed'] and len(acceptance['checks']) == 4
    names = ['bf16', 'attention', 'attention-long', 'graph', 'context',
             'fp32-attention', 'fp32-experts', 'delta']
    checks = 0
    logs = []
    for name in names:
        path = ROOT / ('artifacts/tuned-final-' + name + '.log')
        counts = re.findall(r'^PASS: (\d+)', path.read_text(), flags=re.M)
        assert counts, path
        checks += sum(map(int, counts))
        logs.append(dict(path=str(path), sha256=sha(path)))
    assert checks == 511, checks
    unit = ROOT / 'artifacts/tuned-final-unit.log'
    rpc = ROOT / 'artifacts/tuned-final-rpc.log'
    assert re.search(r'Passed:\s+68', unit.read_text())
    assert '25 integration checks passed' in rpc.read_text()
    before_profile, bp = read('qwen36-hybrid-latest-profile')
    after_profile, ap = read('qwen36-tuned-final-profile')
    result = dict(generatedAtUtc=datetime.now(timezone.utc).isoformat(), rows=rows,
        configuration=dict(expertRows=32, expertRouteTile=64, expertInnerTile=64,
            gateSimdgroups=2, downSimdgroups=1, attentionQueryTokens=4, attentionKeyTokens=128,
            prefillAttentionSimdgroups=16, tinyOrDecodeAttentionSimdgroups=4,
            wideAttentionMinimumQueries=32, mathMode='safe'),
        validation=dict(unitTests=68, nativeOperatorChecks=checks, processRpcChecks=25,
            realModelAcceptanceChecks=4, acceptanceEvidence=acceptance_evidence, nativeLogs=logs),
        diagnosticProfiles=dict(before=bp, after=ap, beforeStages=before_profile['stages'],
            afterStages=after_profile['stages'], warning=after_profile['warning']),
        limitations=[
            'One excluded complete warmup, three measured requests, batch one, fixed 128 outputs with EOS ignored; engines run serially without prefix reuse.',
            'Own residuals, normalization, DeltaNet and decode GEMV remain FP32; BF16 GEMM/attention operands and BF16 KV differ from MLX precision. No broad quality or output-ID parity claim.',
            'Own prefill excludes a small first-emission cost; full first-token timings are retained for both engines.',
            'Prefill remains slower than this installed MLX baseline. No prefill parity claim.',
            'No new full-262144-token model generation or multi-thousand-output benchmark; analytic operator tests reach the final cache address.',
            'Split-command diagnostic timings perturb scheduling and cache residency and are not production throughput.',
            'Feature-split attention, static padded expert operands and static interior attention branches regressed and were reverted. Relaxed math had no consistent benefit and was not selected.'
        ])
    path = OUT / 'qwen36-tuned-summary.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for row in rows:
        print(json.dumps({k: row[k] for k in ['inputTokens', 'ownPromptTokensPerSecond',
            'mlxPromptTokensPerSecond', 'prefillTimeReduction', 'prefillTimeRatioToMlx']}, indent=2))
    print(path)


if __name__ == '__main__':
    main()
