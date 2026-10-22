#!/usr/bin/env python3
"""Audit the separate AC-powered attention dispatch-batch rerun."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/baselines'
spec = importlib.util.spec_from_file_location('attention_audit', ROOT/'scripts/summarize-qwen36-attention-batch.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)
rows = []
for length in [8192, 32768]:
    group = []
    for batch in [128, 512, 1024]:
        row = audit_module.audit(batch, length, 'ac', 3, warmups=3)
        report = json.loads(Path(row['report']).read_text())
        case, = report['cases']
        assert report['requireAcPower']
        observations = list(case['memorySamples'])
        for trial in case['warmups'] + case['trials']:
            observations.extend([trial['powerBefore'], trial['powerAfter']])
        assert all("Now drawing from 'AC Power'" in (s['power'] or '') for s in observations)
        row['acPowerObservations'] = len(observations)
        row['meanDecodeTokensPerSecond'] = statistics.mean(t['timings']['decodeTokensPerSecond'] for t in case['trials'])
        row['prefillTokensPerSecondFromMeanTime'] = length*1000/row['meanPrefillMs']
        group.append(row)
    control = group[0]
    for row in group:
        for field in ['promptSha256', 'promptTokenIdsSha256', 'outputTokenIdsSha256',
                      'precision', 'executableSha256', 'librarySha256', 'commandBufferSubmissions']:
            assert row[field] == control[field], field
        row['prefillTimeChangeFrom128'] = row['meanPrefillMs']/control['meanPrefillMs']-1
        row['extraAllocatedMiBFrom128'] = (row['allocatedBytes']-control['allocatedBytes'])/2**20
    rows.extend(group)
accept_path = OUT/'qwen36-attention-batch-ac-acceptance.json'
acceptance = json.loads(accept_path.read_text())
assert acceptance['passed'] and len(acceptance['checks']) == 4
validation_path = OUT/'qwen36-attention-batch-summary.json'
summary = dict(generatedAtUtc=datetime.now(timezone.utc).isoformat(),
    power='AC required before/after each request and checked at five-second samples during requests',
    warmupsPerCase=3, measuredTrialsPerCase=3, fixedOutputTokens=128,
    selectedBatch=128, outerPrefillChunkTokens=2048, computeQueryTokens=4,
    keyTileTokens=128, kvPartitionTokens=1024, precisionChanged=False, rows=rows,
    powerSettingsObservation=json.loads((OUT/'qwen36-attention-batch-ac-power.json').read_text()),
    restoredDefaultRealModelAcceptance=dict(checks=4, path=str(accept_path), sha256=audit_module.sha(accept_path)),
    operatorAndProcessValidation=dict(path=str(validation_path), sha256=audit_module.sha(validation_path)),
    limitations=['AC attachment is observed, GPU frequency and energy use are not measured.',
        'Sequential batches can still be affected by time, background load, and temperature.',
        'These measurements do not constitute a fresh MLX comparison.',
        'Metal allocations are buffer sizes, not a physical residency or swap attribution measurement.'])
path = OUT/'qwen36-attention-batch-ac-summary.json'
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
for row in rows:
    print(json.dumps({k: row[k] for k in ['inputTokens','batch','prefillMs','meanPrefillMs',
        'prefillTimeChangeFrom128','meanDecodeTokensPerSecond','extraAllocatedMiBFrom128','acPowerObservations']}))
print(path)
