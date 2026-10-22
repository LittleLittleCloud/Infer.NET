#!/usr/bin/env python3
"""Audit same-prompt own/MLX measurements and retain the remaining performance gap."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/baselines'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def own(stem):
 path=OUT/(stem+'.json');j=json.loads(path.read_text());c,=j['cases'];assert c['status']=='completed'
 assert j['excludedWarmupRequests']==1 and len(c['trials'])==3 and len(c['warmups'])==1
 samples=c['trials'];ids=[]
 for sample in samples:
  raw=Path(sample['rawResultFile']);assert sha(raw)==sample['rawResultSha256'];ids.append(json.loads(raw.read_text())['tokenIds'])
 return dict(report=str(path),sha256=sha(path),inputTokens=c['inputTokens'],outputs=c['requestedOutputTokens'],
  promptSha256=c['promptSha256'],promptTokenIdsSha256=c['promptTokenIdsSha256'],
  prefillMs=[s['timings']['prefillMs'] for s in samples],meanPrefillMs=statistics.mean(s['timings']['prefillMs'] for s in samples),
  meanFirstTokenMs=statistics.mean(s['timings']['firstTokenMs'] for s in samples),
  decodeTokensPerSecond=sum(s['timings']['decodeTokens'] for s in samples)*1000/sum(s['timings']['decodeMs'] for s in samples),
  kvBytes=samples[0]['timings']['kvCacheBytes'],metalAllocatedBytes=samples[0]['timings']['metalAllocatedBytes'],
  precision=j['precision'],keyTile=j.get('attentionKeyTileTokens',128),
  executableSha256=j['nativeExecutableSha256'],librarySha256=j['nativeMetalLibrarySha256'],
  identicalOutputIdsWithinVariant=all(value==ids[0] for value in ids),
  excludedWarmupMs=c['warmups'][0]['timings']['prefillMs'])
def mlx(stem):
 path=OUT/(stem+'.json');j=json.loads(path.read_text());assert j.get('completedAtUtc');r,=j['rows'];assert len(r['samples'])==3 and len(r['warmups'])==1
 return dict(report=str(path),sha256=sha(path),inputTokens=r['promptTokens'],outputs=j['requestedGenerationTokens'],
  promptSha256=r['promptSha256'],promptTokenIdsSha256=r['promptTokenIdsSha256'],precision=j['floatPrecision'],
  meanPrefillMs=statistics.mean(s['mlxPromptMs'] for s in r['samples']),meanFirstTokenMs=statistics.mean(s['ttftMs'] for s in r['samples']),
  decodeTokensPerSecond=r['summary']['decodeTokensPerSecond'],versions=j['versions'],revision=j['revision'])
control=own('qwen36-final-control8k');small=own('qwen36-key128-release8k');large=own('qwen36-key128-release32k')
mlsmall=mlx('qwen36-hybrid-mlx8k');mllarge=mlx('qwen36-hybrid-mlx32k')
exe=ROOT/'artifacts/native/osx-arm64/deepinfer';lib=exe.parent/'libdeepinfer_metal.dylib'
for row in [control,small,large]:assert row['executableSha256']==sha(exe) and row['librarySha256']==sha(lib)
comparisons=[]
for mine,external in [(small,mlsmall),(large,mllarge)]:
 for field in ['promptSha256','promptTokenIdsSha256','inputTokens','outputs']:assert mine[field]==external[field],field
 comparisons.append(dict(inputTokens=mine['inputTokens'],ownPromptTokensPerSecond=mine['inputTokens']*1000/mine['meanPrefillMs'],mlxPromptTokensPerSecond=external['inputTokens']*1000/external['meanPrefillMs'],
  prefillTimeRatio=mine['meanPrefillMs']/external['meanPrefillMs'],ownDecodeTokensPerSecond=mine['decodeTokensPerSecond'],mlxDecodeTokensPerSecond=external['decodeTokensPerSecond'],decodeSpeedRatio=mine['decodeTokensPerSecond']/external['decodeTokensPerSecond']))
accept=OUT/'qwen36-key128-acceptance.json';assert json.loads(accept.read_text())['passed']
summary=dict(generatedAtUtc=datetime.now(timezone.utc).isoformat(),own8k=small,own32k=large,control8k=control,mlx8k=mlsmall,mlx32k=mllarge,
 comparisons=comparisons,speedupAgainstPrevious8k=control['meanPrefillMs']/small['meanPrefillMs'],
 acceptanceReport=str(accept),acceptanceSha256=sha(accept),
 validation=dict(unitTests=68,operatorChecks=447,actualModelAcceptanceChecks=4),
 limitations=['One excluded full warmup and three fixed-128-output trials per engine and case. No prefix reuse; model loading outside timing.',
 'Own mixed precision is not identical to MLX BF16. Generated IDs can differ; operator/reference and 8K needle checks are not broad quality equivalence.',
 'Own prefill excludes small first-emission overhead; first-token wall measurements are retained for both engines.',
 'Prefill remains behind MLX; no prefill parity claim.',
 'No new full-262144-token model generation or multi-thousand-token output benchmark. Analytic operator testing reaches the last native address.',
 'Stage profile timings perturb execution; final throughput uses ordinary unsplit commands.',
 'Earlier register-only dequantization and eight-query attention experiments regressed and were not selected.'])
path=OUT/'qwen36-hybrid-summary.json';path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(comparisons=comparisons,speedup8k=summary['speedupAgainstPrevious8k']),indent=2));print(path)
