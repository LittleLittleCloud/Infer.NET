#!/usr/bin/env python3
"""Audit expert-flow trials, binary/raw identity, dispatch changes and power state."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/baselines'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def audit(stem, binary):
    path=OUT/(stem+'.json');j=json.loads(path.read_text());c,=j['cases']
    assert c['status']=='completed' and j['excludedWarmupRequests']==3
    assert len(c['warmups'])==len(c['trials'])==3
    assert j['prefillChunkTokens']==2048 and j['attentionQueryTileTokens']==128
    assert j['nativeExecutableSha256']==sha(binary/'deepinfer')
    assert j['nativeMetalLibrarySha256']==sha(binary/'libdeepinfer_metal.dylib')
    ids=[];submissions=[];dispatches=[];observations=list(c['memorySamples'])
    for sample in c['warmups']+c['trials']:
        raw=Path(sample['rawResultFile']);assert sha(raw)==sample['rawResultSha256']
        r=json.loads(raw.read_text());assert not r['isSimulation'] and len(r['tokenIds'])==128 and r['promptTokens']==c['inputTokens']
        assert sample['timings']['decodeTokens']==127
        ids.append(r['tokenIds']);submissions.append(r['gpuSubmissions']);dispatches.append(r['gpuDispatches'])
        observations += [sample['powerBefore'],sample['powerAfter']]
    assert all(value==ids[0] for value in ids)
    assert len(set(submissions))==len(set(dispatches))==1
    sources=set(s['power'].splitlines()[0] for s in observations)
    settings=set(s['powerSettings'] for s in observations)
    assert len(sources)==len(settings)==1
    times=[s['timings']['prefillMs'] for s in c['trials']]
    return dict(report=str(path),reportSha256=sha(path),inputTokens=c['inputTokens'],
        flow=j['expertFlow'],pipeline=j['expertPipeline'],precision=j['precision'],
        executableSha256=j['nativeExecutableSha256'],librarySha256=j['nativeMetalLibrarySha256'],
        promptSha256=c['promptSha256'],promptTokenIdsSha256=c['promptTokenIdsSha256'],
        outputTokenIdsSha256=hashlib.sha256(json.dumps(ids[0],separators=(',',':')).encode()).hexdigest(),
        prefillMs=times,meanPrefillMs=statistics.mean(times),sampleStddevMs=statistics.stdev(times),
        decodeTokensPerSecond=381000/sum(s['timings']['decodeMs'] for s in c['trials']),
        allocatedBytes=c['trials'][0]['timings']['metalAllocatedBytes'],
        submissions=submissions[0],dispatches=dispatches[0],powerSource=next(iter(sources)),
        powerSettings=next(iter(settings)),powerObservations=len(observations))

def compare(rows):
    control=rows[0]
    for row in rows:
        for key in ['precision','executableSha256','librarySha256','inputTokens','promptSha256',
                    'promptTokenIdsSha256','outputTokenIdsSha256','submissions','allocatedBytes','powerSource','powerSettings']:
            assert row[key]==control[key],key
        expected=control['dispatches']-(40*(row['inputTokens']//2048) if row['flow']=='direct' else 0)
        assert row['dispatches']==expected
        row['prefillTimeReduction']=1-row['meanPrefillMs']/control['meanPrefillMs']
    return rows

pilot=compare([audit('qwen36-expert-flow-'+name+'-8k-highpower',ROOT/'artifacts/native/expert-flow-first')
    for name in ['staged-serial','direct-serial','direct-double-buffer']])
confirmation=[]
for length in [8,32]:
    confirmation += compare([audit(f'qwen36-expert-flow-{label}-{length}k-confirm',ROOT/'artifacts/native/osx-arm64')
        for label in ['control','direct']])
recheck=audit('qwen36-expert-flow-direct-32k-recheck',ROOT/'artifacts/native/osx-arm64')
control32=next(r for r in confirmation if r['inputTokens']==32768 and r['flow']=='staged')
compare([control32,recheck])
for length in [8192,32768]:
    matched=[r for r in pilot+confirmation+[recheck] if r['inputTokens']==length]
    for key in ['precision','promptSha256','promptTokenIdsSha256','outputTokenIdsSha256']:
        assert all(r[key]==matched[0][key] for r in matched),key
evidence=[];native_checks=0
for label in ['serial','pipeline','fp32','bf16']:
    path=ROOT/f'artifacts/expert-flow-final-{label}.log'
    native_checks+=sum(map(int,re.findall(r'^PASS: (\d+)',path.read_text(),flags=re.M)))
    evidence.append(dict(path=str(path),sha256=sha(path)))
assert native_checks==146
for label,pattern in [('unit',r'Passed:\s+68'),('rpc',r'25 integration checks passed')]:
    path=ROOT/f'artifacts/expert-flow-final-{label}.log'
    assert re.search(pattern,path.read_text())
    evidence.append(dict(path=str(path),sha256=sha(path)))
accept_path=OUT/'qwen36-expert-flow-acceptance.json'
acceptance=json.loads(accept_path.read_text())
assert acceptance['passed'] and len(acceptance['checks'])==4
evidence.append(dict(path=str(accept_path),sha256=sha(accept_path)))
result=dict(generatedAtUtc=datetime.now(timezone.utc).isoformat(),selectedFlow='staged',candidateFlow='direct',selectedPipeline='serial',
    conclusion='Keep the old default: direct output slightly improves 8K but has not shown a 32K benefit; double buffering regresses. Both alternatives remain explicit diagnostic options.',
    direct32kRecheck=recheck,
    validation=dict(nativeChecks=146,unitTests=68,processRpcChecks=25,directFlowRealModelChecks=4,evidence=evidence),
    pilot=pilot,confirmation=confirmation,confirmationOrder={'8k':['control','direct'],'32k':['direct','control']},
    precisionChanged=False,weightsChanged=False,prefixReuse=False,excludedWarmupsPerCase=3,measuredTrialsPerCase=3,
    limitations=['Power mode/source is recorded; GPU clocks and hardware occupancy were not measured.',
        'Between-round timing drift exists. Compare within each same-binary round only.',
        'Fixed 128-token output is an EOS-ignored throughput diagnostic; it does not establish long-output quality.',
        'No new MLX baseline or parity claim.',
        'Experimental cooperative BF16 tensor-store layout was rejected by exact-output checks; it is absent from the selected kernel.',
        'Static double buffering also regressed isolated chain timings and was removed; the explicit double-buffer option remains diagnostic only.'])
path=OUT/'qwen36-expert-flow-summary.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for row in pilot+confirmation:
    print(json.dumps({k:row[k] for k in ['inputTokens','flow','pipeline','meanPrefillMs','prefillTimeReduction','decodeTokensPerSecond','dispatches']}))
print(path)
