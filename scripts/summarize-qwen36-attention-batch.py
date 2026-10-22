#!/usr/bin/env python3
"""Audit same-binary query dispatch-batch experiments without claiming a speedup."""
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


def audit(batch, length, tag, trials, warmups=1):
    path = OUT / f'qwen36-ab{batch}-{length//1024}k-{tag}.json'
    j = json.loads(path.read_text())
    c, = j['cases']
    assert c['status'] == 'completed' and len(c['warmups']) == warmups
    assert len(c['trials']) == trials and j['excludedWarmupRequests'] == warmups
    assert j['attentionQueryTileTokens'] == batch and j['prefillChunkTokens'] == 2048
    assert j['attentionComputeQueryTokens'] == 4 and j['attentionKeyTileTokens'] == 128
    published = ROOT / 'artifacts/native/osx-arm64'
    assert j['nativeExecutableSha256'] == sha(published/'deepinfer')
    assert j['nativeMetalLibrarySha256'] == sha(published/'libdeepinfer_metal.dylib')
    ids, dispatches, submissions = [], [], []
    for sample in c['warmups'] + c['trials']:
        raw_path = Path(sample['rawResultFile'])
        assert sha(raw_path) == sample['rawResultSha256']
        raw = json.loads(raw_path.read_text())
        assert not raw['isSimulation'] and raw['promptTokens'] == length
        assert len(raw['tokenIds']) == 128 and sample['timings']['decodeTokens'] == 127
        ids.append(raw['tokenIds'])
        dispatches.append(raw['gpuDispatches'])
        submissions.append(raw['gpuSubmissions'])
    assert all(value == ids[0] for value in ids)
    assert len(set(dispatches)) == len(set(submissions)) == 1
    times = [s['timings']['prefillMs'] for s in c['trials']]
    return dict(report=str(path), reportSha256=sha(path), batch=batch, inputTokens=length,
        promptSha256=c['promptSha256'], promptTokenIdsSha256=c['promptTokenIdsSha256'],
        outputTokenIdsSha256=hashlib.sha256(json.dumps(ids[0],separators=(',',':')).encode()).hexdigest(),
        precision=j['precision'], executableSha256=j['nativeExecutableSha256'],
        librarySha256=j['nativeMetalLibrarySha256'], prefillMs=times,
        meanPrefillMs=statistics.mean(times), sampleStddevMs=statistics.stdev(times) if trials>1 else None,
        kernelDispatches=dispatches[0], commandBufferSubmissions=submissions[0],
        allocatedBytes=c['trials'][0]['timings']['metalAllocatedBytes'],
        excludedWarmupMs=[s['timings']['prefillMs'] for s in c['warmups']])


def main():
    pilot, confirm = [], []
    for length in [8192,32768]:
        pilot += [audit(batch,length,'scout',1) for batch in [128,256,512,1024,2048]]
        rows = [audit(batch,length,'confirm',3) for batch in [128,512,1024]]
        control = rows[0]
        for row in rows:
            for field in ['promptSha256','promptTokenIdsSha256','outputTokenIdsSha256','precision',
                          'executableSha256','librarySha256','commandBufferSubmissions']:
                assert row[field] == control[field], field
            row['prefillTimeChangeFrom128'] = row['meanPrefillMs']/control['meanPrefillMs']-1
            row['extraAllocatedMiBFrom128'] = (row['allocatedBytes']-control['allocatedBytes'])/2**20
        confirm += rows
    for row in pilot:
        control = next(r for r in confirm if r['inputTokens']==row['inputTokens'] and r['batch']==128)
        for field in ['promptSha256','promptTokenIdsSha256','outputTokenIdsSha256','precision',
                      'executableSha256','librarySha256','commandBufferSubmissions']:
            assert row[field] == control[field], field
    settled = [audit(batch,32768,'settled',3,warmups=3) for batch in [128,512]]
    for row in settled:
        for field in ['promptSha256','promptTokenIdsSha256','outputTokenIdsSha256','precision',
                      'executableSha256','librarySha256','commandBufferSubmissions']:
            assert row[field] == settled[0][field]
        row['prefillTimeChangeFrom128'] = row['meanPrefillMs']/settled[0]['meanPrefillMs']-1
    native_logs = ['checks','oracle','max-context','default-context','fp32-context']
    checks = 0
    evidence = []
    for label in native_logs:
        path = ROOT / f'artifacts/attention-batch-{label}.log'
        counts = re.findall(r'^PASS: (\d+)', path.read_text(), flags=re.M)
        assert counts
        checks += sum(map(int,counts))
        evidence.append(dict(path=str(path),sha256=sha(path)))
    assert checks == 157
    accept_path = OUT/'qwen36-attention-batch-acceptance.json'
    acceptance = json.loads(accept_path.read_text())
    assert acceptance['passed'] and len(acceptance['checks']) == 4
    unit = ROOT/'artifacts/attention-batch-unit.log'
    rpc = ROOT/'artifacts/attention-batch-rpc.log'
    assert re.search(r'Passed:\s+68',unit.read_text())
    assert '25 integration checks passed' in rpc.read_text()
    result = dict(generatedAtUtc=datetime.now(timezone.utc).isoformat(),
        conclusion='No broad robust speedup justifies larger scratch allocation; retain 128. Small case-specific differences and between-run drift are retained below.',
        selectedBatch=128, outerPrefillChunkTokens=2048, computeQueryTokens=4,
        keyTileTokens=128, kvPartitionTokens=1024, precisionChanged=False,
        scoutOrder=[128,256,512,1024,2048], confirmationOrder=[1024,512,128],
        excludedWarmupsPerCase=1, measuredTrialsPerConfirmation=3,
        fixedOutputTokens=128, prefixReuse=False, pilot=pilot, confirmation=confirm,
        extraWarmupRetest=dict(order=[128,512],excludedWarmups=3,measuredTrials=3,rows=settled),
        validation=dict(nativeChecks=checks,unitTests=68,processRpcChecks=25,realModelAcceptanceChecks=4,
            nativeEvidence=evidence,acceptance=dict(path=str(accept_path),sha256=sha(accept_path))),
        mlxSource=json.loads((OUT/'qwen36-attention-batch-mlx-source.json').read_text()),
        powerObservation=json.loads((OUT/'qwen36-attention-batch-power.json').read_text()),
        limitations=[
            'These are dispatch batches, distinct from the four-query compute tile, 128-key inner tile and 1024-key reduction partition.',
            'The pilot and confirmation rounds show between-run drift; single-sample pilot differences are not a speedup or regression claim.',
            'Battery power was observed during confirmation, but power/clocks were not captured continuously. Do not compare absolute times to the earlier MLX/own timing round or attribute drift to one cause.',
            'No fresh MLX inference benchmark or new MLX parity claim. The expected M5 kernel geometry is inferred from tagged upstream source, not a Metal capture.',
            'Allocated Metal buffer bytes do not establish physical residency or swap attribution.',
            'Large-query max-window attention has analytic operator coverage, not a new full-window whole-model generation benchmark.'
        ])
    path = OUT/'qwen36-attention-batch-summary.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for row in confirm:
        print(json.dumps({k:row[k] for k in ['inputTokens','batch','meanPrefillMs',
            'prefillTimeChangeFrom128','extraAllocatedMiBFrom128','kernelDispatches','commandBufferSubmissions']},indent=2))
    print(path)


if __name__ == '__main__':
    main()
