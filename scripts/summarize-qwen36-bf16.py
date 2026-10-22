#!/usr/bin/env python3
"""Summarize pinned-prompt mixed precision measurements without claiming model quality parity."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/baselines'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def report(name):
    path = OUT/(name+'.json')
    data = json.loads(path.read_text())
    case, = data['cases']
    assert case['status'] == 'completed', (name, case['status'])
    raw = [json.loads(Path(t['rawResultFile']).read_text()) for t in case['trials']]
    for t in case['trials']:
        assert sha(Path(t['rawResultFile'])) == t['rawResultSha256']
    result = dict(reportFile=str(path), reportSha256=sha(path),
                  promptSha256=case['promptSha256'], inputTokens=case['inputTokens'],
                  requestedOutputTokens=case['requestedOutputTokens'], trials=len(raw),
                  prefillSeconds=[t['timings']['prefillMs']/1000 for t in case['trials']],
                  precision=data['precision'],
                  decodeTokensPerSecond=sum(t['timings']['decodeTokens'] for t in case['trials']) /
                    (sum(t['timings']['decodeMs'] for t in case['trials'])/1000),
                  kvBytes=case['trials'][0]['timings']['kvCacheBytes'],
                  metalAllocatedBytes=case['trials'][0]['timings']['metalAllocatedBytes'],
                  nativeExecutableSha256=data['nativeExecutableSha256'],
                  nativeMetalLibrarySha256=data['nativeMetalLibrarySha256'],
                  deterministicIds=all(r['tokenIds'] == raw[0]['tokenIds'] for r in raw),
                  generatedText=raw[0]['text'])
    result['meanPrefillSeconds'] = statistics.mean(result['prefillSeconds'])
    result['promptTokensPerSecond'] = result['inputTokens']/result['meanPrefillSeconds']
    return result, raw[0]['tokenIds']

control, cids = report('qwen36-bf16-control8k')
release, rids = report('qwen36-bf16-release8k')
large, lids = report('qwen36-bf16-release32k')
historical, hids = report('qwen36-grouped-delta-release32k')
assert control['promptSha256'] == release['promptSha256']
assert large['promptSha256'] == historical['promptSha256']
exe = ROOT/'artifacts/native/osx-arm64/deepinfer'
lib = exe.parent/'libdeepinfer_metal.dylib'
for current in [control, release, large]:
    assert current['nativeExecutableSha256'] == sha(exe)
    assert current['nativeMetalLibrarySha256'] == sha(lib)
acceptance = OUT/'qwen36-bf16-acceptance.json'
assert json.loads(acceptance.read_text())['passed']
summary = dict(generatedAtUtc=datetime.now(timezone.utc).isoformat(),
               control8k=control, release8k=release, release32k=large,
               historicalControl32k=historical,
               speedup8k=control['meanPrefillSeconds']/release['meanPrefillSeconds'],
               historicalSpeedup32k=historical['meanPrefillSeconds']/large['meanPrefillSeconds'],
               identicalOutputIds8k=cids == rids,
               firstDifferentOutput8k=next((i+1 for i,(a,b) in enumerate(zip(cids,rids)) if a != b), None),
               identicalOutputIds32k=lids == hids,
               acceptanceReport=str(acceptance), acceptanceSha256=sha(acceptance),
               validation=dict(unitTests=68, nativeFp32Checks=210, nativeBf16Checks=136,
                               rpcIntegrationChecks=25, longModelAcceptanceChecks=4,
                               independentFp32TraceComparison=str(ROOT/'artifacts/reference/qwen36-bf16-fp32-regression/comparison.json')),
               limitations=[
                   'Mixed precision: GEMM operands and KV only; activation buffers, state, attention arithmetic and decode GEMV remain FP32.',
                   'No excluded warmup; all saved prompt inputs use fixed 128 output tokens with EOS ignored.',
                   '32K comparison is to one historical FP32 trial, not a new repeated paired comparison.',
                   'Kernel oracles and 8K needle/short checks are not broad model quality or MLX equivalence evidence.',
                   'The new BF16 path has not rerun the full 262144-token window or multi-thousand-token output test.',
                   'No new matched-precision/long-input MLX comparison; no parity claim.',
                   'The initial scalar-unpack 8K experiment is exploratory; its third trial overlapped small hardware tests and is not used in the final comparison.'
               ])
path = OUT/'qwen36-bf16-summary.json'
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: summary[k] for k in ['speedup8k','historicalSpeedup32k','identicalOutputIds8k','firstDifferentOutput8k','identicalOutputIds32k']}, indent=2))
print(path)
