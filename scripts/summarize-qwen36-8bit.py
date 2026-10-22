#!/usr/bin/env python3
"""Summarize measured 8-bit trials and capacity-only two-device candidates."""
import json
import hashlib
from pathlib import Path
import re
import statistics
import struct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/baselines'
model = ROOT/'models/qwen3.6-35b-a3b-8bit-mlx'
current_library_hash = hashlib.sha256((ROOT/'artifacts/native/qwen36-8bit/libdeepinfer_metal.dylib').read_bytes()).hexdigest()
sizes = {}
for path in sorted(model.glob('*.safetensors')):
    with path.open('rb') as f:
        length, = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(length))
    for name,t in header.items():
        if name == '__metadata__': continue
        assert name not in sizes
        sizes[name] = t['data_offsets'][1]-t['data_offsets'][0]
layers = [sum(n for k,n in sizes.items() if k.startswith(f'language_model.model.layers.{i}.')) for i in range(40)]
language = sum(n for k,n in sizes.items() if k.startswith('language_model.'))
embed = sum(n for k,n in sizes.items() if k.startswith('language_model.model.embed_tokens.'))
assert language > 30*2**30 and all(layers)
summary = {'model':json.loads((model/'provenance.json').read_text())['repository'],
           'currentNativeLibrarySha256':current_library_hash,
           'languageWeightBytes':language,'layerWeightBytes':layers,'embeddingBytes':embed,
           'measurements':[],'pipelineCandidates':[],
           'limits':['CUDA stage throughput and network not measured',
                     'Cross-request prefix cache not implemented; repeated requests are cold-state prefill',
                     'Pipeline times below are Mac-stage proportional lower-bound estimates, not end-to-end measurements']}
def page_count(snapshot, name):
    m = re.search(r'^'+name+r':\s+(\d+)',snapshot.get('vm',''),re.M)
    return int(m[1]) if m else None
def suffix_proxies(file, tokens, warmups, trials):
    """Late chunks of cold prefill are a compute proxy, NOT a cache-hit benchmark."""
    log = file.with_name(file.stem+'-cluster.log')
    if not log.exists(): return []
    runs, points = [], [(0,0.0)]
    for done,total,ms in re.findall(r'PREFILL_PROGRESS tokens=(\d+)/(\d+) elapsedMs=([\d.]+)', log.read_text()):
        done,total,seconds = int(done),int(total),float(ms)/1000
        if total != tokens: continue
        points.append((done,seconds))
        if done == total:
            threshold = total*.9
            for (x0,y0),(x1,y1) in zip(points,points[1:]):
                if x0 <= threshold <= x1:
                    runs.append(seconds-(y0+(y1-y0)*(threshold-x0)/(x1-x0)))
                    break
            points = [(0,0.0)]
    return runs[warmups:warmups+trials]
for file in sorted(OUT.glob('qwen36-8bit-*k-*out.json')):
    report = json.loads(file.read_text())
    for case in report['cases']:
        if case['status'] != 'completed': continue
        samples = case['trials']
        ids = [json.loads(Path(t['rawResultFile']).read_text())['tokenIds'] for t in case.get('warmups',[])+samples]
        row = {'inputTokens':case['inputTokens'],'outputTokens':case['requestedOutputTokens'],
            'prefillChunkTokens':report['prefillChunkTokens'],
            'nativeMetalLibrarySha256':report['nativeMetalLibrarySha256'],
            'matchesCurrentNativeLibrary':report['nativeMetalLibrarySha256']==current_library_hash,
            'trials':len(samples),'excludedWarmups':len(case.get('warmups',[])),
            'warmupPrefillSeconds':[t['timings']['prefillMs']/1000 for t in case.get('warmups',[])],
            'prefillSeconds':statistics.mean(t['timings']['prefillMs']/1000 for t in samples),
            'firstTokenSeconds':statistics.mean(t['timings']['firstTokenMs']/1000 for t in samples),
            'decodeTokensPerSecond':sum(t['actualOutputTokens']-1 for t in samples)*1000/sum(t['timings']['decodeMs'] for t in samples),
            'maxMetalAllocatedGiB':max(t['timings']['metalAllocatedBytes']/2**30 for t in samples),
            'contextCapacity':samples[0]['timings']['contextCapacity'],
            'outputSequencesComparedIncludingWarmup':len(ids),
            'outputIdsRepeatable':all(t==ids[0] for t in ids) if len(ids)>1 else None,'report':str(file),
            'systemSwapPages':[{k:page_count(t['powerAfter'],k)-page_count(t['powerBefore'],k) for k in ['Swapins','Swapouts']} for t in samples]}
        summary['measurements'].append(row)
        proxies = suffix_proxies(file,case['inputTokens'],len(case.get('warmups',[])),len(samples))
        if proxies:
            row['last10PercentPrefillSecondsProxy']=statistics.mean(proxies)
N=131072
delta_per_linear=32*128*128*4+3*8192*4
for count in [20,24,28]:
    full=count//4;linear=count-full
    weight=embed+sum(layers[:count])
    state=N*2048*full+linear*delta_per_linear
    wire=state+N*2048*4
    entry={'frontLayers':count,'macTailLayers':40-count,'frontWeightGiB':weight/2**30,
           'front128kStateGiB':state/2**30,'frontWeightAndStateGiB':(weight+state)/2**30,
           'remainingOf24GiB':24-(weight+state)/2**30,
           'wireGiB':wire/2**30,'wireSecondsAt110MBps':wire/110e6,'wireSecondsAt1GBps':wire/1e9}
    for row in summary['measurements']:
        if row['inputTokens']==N and row['outputTokens']==128 and row['matchesCurrentNativeLibrary']:
            entry['macStageProportionalSeconds']=row['prefillSeconds']*(40-count)/40
            work=2*2436710400*N+81920*N*(N+1)
            entry['projectionAndAttentionFrontTFlop']=work*(count/40)/1e12
            entry['gpuEffectiveTFlopsToMatchMacTail']=entry['projectionAndAttentionFrontTFlop']/entry['macStageProportionalSeconds']
            entry['estimationBasisReport']=row['report']
            if 'last10PercentPrefillSecondsProxy' in row:
                entry['macTail90PercentPrefixProxySeconds']=row['last10PercentPrefillSecondsProxy']*(40-count)/40
    entry['wireGiBWith90PercentPrefixOnBothDevices']=(N*.1*(8192+2048*full)+linear*delta_per_linear)/2**30
    summary['pipelineCandidates'].append(entry)
target=OUT/'qwen36-8bit-summary.json'
target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
