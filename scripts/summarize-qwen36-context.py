#!/usr/bin/env python3
"""Export only completed long-context trials; preserve unsupported/pending/failed case states separately."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/baselines'
rows, states, windows = [], [], []
for name in ['qwen36-context-fixed', 'qwen36-context-large-fixed', 'qwen36-context-natural', 'qwen36-context-full',
             'qwen36-prefill-final-legacy8k', 'qwen36-prefill-final-fused8k',
             'qwen36-prefill-final-fused32k', 'qwen36-context-fused-full',
             'qwen36-grouped-delta-final-control','qwen36-grouped-delta-final-sequential',
             'qwen36-grouped-delta-final-chunked','qwen36-grouped-delta-release8k',
             'qwen36-grouped-delta-release32k', 'qwen36-bf16-control8k',
             'qwen36-bf16-release8k', 'qwen36-bf16-release32k',
             'qwen36-final-control8k','qwen36-key128-release8k','qwen36-key128-release32k',
             'qwen36-tuned-release8k','qwen36-tuned-release32k',
             *[f'qwen36-ab{batch}-{length}k-confirm' for length in [8,32] for batch in [128,512,1024]],
             'qwen36-ab128-32k-settled','qwen36-ab512-32k-settled',
             *[f'qwen36-ab{batch}-{length}k-ac' for length in [8,32] for batch in [128,512,1024]],
             *[f'qwen36-expert-flow-{name}-8k-highpower' for name in ['staged-serial','direct-serial','direct-double-buffer']],
             *[f'qwen36-expert-flow-{name}-{length}k-confirm' for length in [8,32] for name in ['control','direct']],
             'qwen36-expert-flow-direct-32k-recheck']:
    path = OUT/(name+'.json')
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    precision = report.get('precision', {})
    if not isinstance(precision, dict): precision = {}
    for case in report['cases']:
        states.append(dict(report=name, inputBudget=case['inputBudget'], requestedOutputTokens=case['requestedOutputTokens'], status=case['status']))
        for i, trial in enumerate(case.get('trials', []), 1):
            t = trial['timings']; ws = t['decodeWindows']
            row = dict(report=name, trial=i, mode=report['mode'], inputTokens=case['inputTokens'],
                       prefillChunkTokens=report.get('prefillChunkTokens',32),
                       prefillAttention=report.get('prefillAttention','legacy'),
                       prefillExperts=report.get('prefillExperts','vector'),
                       prefillDelta=report.get('prefillDelta','sequential'),
                       prefillGemm=precision.get('prefillGemm','fp32'),
                       kvPrecision=precision.get('kv','fp32'),
                       gemmImpl=precision.get('gemmImpl','staged'),
                       expertImpl=precision.get('expertImpl','staged'),
                       attentionImpl=precision.get('attentionImpl','fp32'),
                       attentionQueryBatchTokens=report.get('attentionQueryTileTokens',128),
                       expertFlow=report.get('expertFlow','staged'),
                       expertPipeline=report.get('expertPipeline','serial'),
                       nativeExecutableSha256=report.get('nativeExecutableSha256',''),
                       nativeMetalLibrarySha256=report.get('nativeMetalLibrarySha256',''),
                       excludedWarmups=report.get('excludedWarmupRequests',0),
                       outputTokens=trial['actualOutputTokens'], finishReason=trial['finishReason'],
                       firstTokenSeconds=t['firstTokenMs']/1000, prefillSeconds=t['prefillMs']/1000,
                       prefillTokensPerSecond=trial['prefillTokensPerSecond'],
                       decodeTokensPerSecond=t['decodeTokensPerSecond'], rpcSeconds=trial['rpcMs']/1000,
                       firstWindowTokensPerSecond=ws[0]['tokensPerSecond'] if ws else 0,
                       lastWindowTokensPerSecond=ws[-1]['tokensPerSecond'] if ws else 0,
                       capacity=t['contextCapacity'], kvGiB=t['kvCacheBytes']/2**30, metalAllocatedGiB=t['metalAllocatedBytes']/2**30,
                       rawResultFile=trial['rawResultFile'], rawResultSha256=trial['rawResultSha256'])
            rows.append(row)
            windows += [dict(report=name, trial=i, inputTokens=case['inputTokens'], outputTokens=trial['actualOutputTokens'], **w) for w in ws]
result = dict(generatedAtUtc=datetime.now(timezone.utc).isoformat(), exploratoryTrials=True,
              warmupPolicy='Per-trial excludedWarmups; historical cases vary', firstTokenIsWorkerInternal=True, trials=rows, cases=states)
(OUT/'qwen36-context-summary.json').write_text(json.dumps(result, indent=2)+'\n')
for filename, data in [('qwen36-context-summary.csv', rows), ('qwen36-context-decode-windows.csv', windows)]:
    if data:
        with (OUT/filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
print(f'Saved {len(rows)} completed trials and {len(windows)} decode windows; {len(states)} case states')
