#!/usr/bin/env python3
"""Audit final same-binary prefill A/B and independent, fresh MLX evidence."""
import hashlib
import json
from pathlib import Path
import re
import statistics
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/baselines'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def mean(xs): return statistics.mean(xs)
def evidence(path): return {'path':str(path),'sha256':sha(path)}
exe=ROOT/'artifacts/native/osx-arm64/deepinfer'
lib=exe.with_name('libdeepinfer_metal.dylib')
summary={'selected':{'expertTiles':'adaptive','expertGate':'fused','deltaImpl':'device',
                    'attentionPartitionTokens':1024,'expertFlow':'staged','expertPipeline':'serial'},
         'nativeExecutable':evidence(exe),'nativeMetalLibrary':evidence(lib),'cases':[],
         'validation':[],'powerObservations':0}
reference_settings=None
for length in [8192,32768]:
    row={'inputTokens':length,'outputTokens':128,'decodeIntervals':127}
    reference_ids=None
    for label in ['control','selected']:
        path=OUT/f'qwen36-joy-{label}-{length//1024}k.json';report=json.loads(path.read_text());case,=report['cases']
        assert case['status']=='completed' and len(case['trials'])==len(case['warmups'])==3
        assert report['nativeExecutableSha256']==sha(exe) and report['nativeMetalLibrarySha256']==sha(lib)
        assert report['attentionPartitionTokens']==1024 and report['expertFlow']=='staged' and report['expertPipeline']=='serial'
        assert (report['expertTiles'],report['expertGate'],report['deltaImpl']) == (('fixed','split','staged') if label=='control' else ('adaptive','fused','device'))
        assert report['prefillChunkTokens']==2048 and report['attentionQueryTileTokens']==128
        assert not report['prefixReuse'] and report['requireAcPower']
        observations=list(case['memorySamples']);raws=[];dispatch=[];submissions=[]
        for trial in case['warmups']+case['trials']:
            observations += [trial['powerBefore'],trial['powerAfter']]
            raw=Path(trial['rawResultFile']);assert sha(raw)==trial['rawResultSha256'];r=json.loads(raw.read_text())
            assert not r['isSimulation'] and r['promptTokens']==length and len(r['tokenIds'])==128
            assert trial['timings']['decodeTokens']==127
            if reference_ids is None: reference_ids=r['tokenIds']
            assert r['tokenIds']==reference_ids, 'A/B output IDs changed'
            raws.append(evidence(raw));dispatch.append(r['gpuDispatches']);submissions.append(r['gpuSubmissions'])
        for sample in observations:
            assert "'AC Power'" in sample['power']
            if reference_settings is None: reference_settings=sample['powerSettings']
            assert sample['powerSettings']==reference_settings
        summary['powerObservations']+=len(observations)
        # Existing swap allocation can persist from prior work; audit activity deltas.
        def counter(sample,key):
            return int(re.search(r'^'+key+r':\s+(\d+)',sample['vm'],re.M)[1])
        activity=[{'swapins':counter(t['powerAfter'],'Swapins')-counter(t['powerBefore'],'Swapins'),
                   'swapouts':counter(t['powerAfter'],'Swapouts')-counter(t['powerBefore'],'Swapouts')}
                  for t in case['trials']]
        row[label]={'report':evidence(path),'rawResults':raws,
                    'prefillMs':[t['timings']['prefillMs'] for t in case['trials']],
                    'decodeTokensPerSecond':[t['timings']['decodeTokensPerSecond'] for t in case['trials']],
                    'ttftMs':[t['timings']['firstTokenMs'] for t in case['trials']],
                    'allocatedMetalBytes':[t['timings']['metalAllocatedBytes'] for t in case['trials']],
                    'dispatches':sorted(set(dispatch)),'submissions':sorted(set(submissions)),
                    'measuredSystemSwapPageDeltas':activity,
                    'promptSha256':case['promptSha256'],'promptTokenIdsSha256':case['promptTokenIdsSha256']}
    assert row['control']['allocatedMetalBytes']==row['selected']['allocatedMetalBytes']
    assert row['control']['dispatches']==row['selected']['dispatches']
    assert row['control']['submissions']==row['selected']['submissions']
    path=OUT/f'qwen36-joy-mlx{length//1024}k.json';report=json.loads(path.read_text());case,=report['rows']
    assert len(case['samples'])==len(case['warmups'])==3
    assert report['prefillStepSize']==2048 and report['floatPrecision']=='bf16'
    assert all(v['matchesInstalledRecord'] for v in report['runtimeModuleHashes'].values())
    provenance=ROOT/'models/qwen3.6-35b-a3b-5bit-mlx/provenance.json'
    assert report['provenanceSha256']==sha(provenance)
    summary['modelProvenance']=evidence(provenance)
    assert "'AC Power'" in report['powerBefore'] and "'AC Power'" in report['powerAfter']
    assert report['powerSettings']==reference_settings
    for k in ['promptSha256','promptTokenIdsSha256']: assert case[k]==row['control'][k]==row['selected'][k]
    assert all(t['generatedTokens']==128 and t['decodeIntervals']==127 for t in case['samples'])
    row['mlx']={'report':evidence(path),'prefillMs':[t['mlxPromptMs'] for t in case['samples']],
                'decodeTokensPerSecond':[t['decodeTokensPerSecond'] for t in case['samples']],
                'ttftMs':[t['ttftMs'] for t in case['samples']]}
    for label in ['control','selected','mlx']:
        r=row[label];r['meanPrefillMs']=mean(r['prefillMs']);r['prefillTokensPerSecond']=length/(r['meanPrefillMs']/1000)
        r['meanDecodeTokensPerSecond']=mean(r['decodeTokensPerSecond'])
    row['prefillTimeReductionPercent']=100*(1-row['selected']['meanPrefillMs']/row['control']['meanPrefillMs'])
    row['ownPrefillLongerThanMlxPercent']=100*(row['selected']['meanPrefillMs']/row['mlx']['meanPrefillMs']-1)
    row['allOwnWarmupAndTrialOutputIdsIdentical']=True
    summary['cases'].append(row)
for name,pattern in [('publish',r'DeepInfer.Host -> .*artifacts/native/osx-arm64/'),
                     ('experts',r'PASS: 56 grouped expert checks'),('flow',r'PASS: 21 expert-flow checks'),
                     ('delta',r'PASS: 60 chunkwise delta checks'),('attention',r'PASS:'),
                     ('attention-long',r'PASS: 12 BF16 attention analytic checks'),
                     ('partition',r'PASS:'),('unit',r'Passed:\s+68'),('rpc',r'25 integration checks passed'),
                     ('acceptance',r'PASS: 4 long RPC acceptance checks')]:
    path=ROOT/'artifacts'/f'joy-final-{name}.log';text=path.read_text();assert re.search(pattern,text), (name,text[-1000:])
    summary['validation'].append(evidence(path))
acceptance=OUT/'qwen36-joy-acceptance.json';assert json.loads(acceptance.read_text())['passed'];summary['acceptance']=evidence(acceptance)
summary['nativeChecks']=sum(sum(map(int,re.findall(r'^PASS: (\d+)',Path(v['path']).read_text(),re.M))) for v in summary['validation'] if Path(v['path']).stem in ['joy-final-experts','joy-final-flow','joy-final-delta','joy-final-attention','joy-final-attention-long','joy-final-partition'])
assert summary['nativeChecks']==287
summary['limitations']=['Batch one; only 8K/32K input with fixed 128 outputs in this round.',
 'MLX prompt timing includes the first generated token; own prefill timing excludes first-emission overhead. TTFT retained.',
 'Own A/B token IDs are identical; cross-engine output/precision equivalence is not asserted.',
 'System swap counters are system-wide, not worker-specific.']
path=OUT/'qwen36-joy-summary.json';path.write_text(json.dumps(summary,indent=2)+'\n')
for r in summary['cases']:
    print(r['inputTokens'], {k:round(r[k]['meanPrefillMs'],3) for k in ['control','selected','mlx']},
          'reduction',round(r['prefillTimeReductionPercent'],3),'MLX gap',round(r['ownPrefillLongerThanMlxPercent'],3))
print('Audited',summary['powerObservations'],'own power observations;',path)
