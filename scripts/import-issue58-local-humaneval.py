#!/usr/bin/env python3
"""Audit and import the separately measured Issue #58 M5 Pro cohort."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from home_trends import import_pi, validate

SHA = '87c2b7dcaf4a418dcb18707e1913562717b43cdc'
BASELINE = 'eaaf81f630614b4d6416bc62776c13d36331fa57'
RUN_ID = 'issue58-m5pro-local-human20-20261024'
CONFIG_ID = 'pi-human20-m5pro-local-affine8-fp32-observed'
REPORT = 'docs/EXPERIMENT-58-M5PRO-HUMAN20-20261024.md'
SOURCE = 'https://github.com/LittleLittleCloud/Infer.NET/blob/main/site/trends.json'


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instant(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def swapouts(sample):
    return int(re.search(r'^Swapouts:\s+(\d+)', sample['vm_stat'], re.M)[1]) * int(
        re.search(r'page size of (\d+) bytes', sample['vm_stat'])[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--results-pr', required=True)
    args = parser.parse_args()
    base = args.run_dir.resolve()
    cohort = base / 'cohort-20'
    manifest, summary = read(cohort/'manifest.json'), read(cohort/'summary.json')
    tasks, observed = lines(cohort/'tasks.jsonl'), lines(base/'cohort-observer/requests.jsonl')
    functional = read(base/'functional.json')
    assert len(tasks) == len(manifest['ids']) == 20
    assert [task['task_id'] for task in tasks] == manifest['ids']
    assert manifest['endpoint_build'] == SHA and manifest['seed'] == 0
    assert manifest['pi_version'] == '1.4.0' and manifest['timeout_seconds'] == 300
    assert manifest['dataset_sha256'] == 'b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef'
    assert manifest['model_revision'] == 'e06a74e6236a60c8367e1a3214e83d8b61b637b0'
    assert manifest['context_window'] == 32768 and manifest['max_tokens'] == 4096
    assert summary['tokens']['cacheRead'] == summary['tokens']['cacheWrite'] == 0
    assert len(observed) == summary['request_count'], 'Every Pi request requires a captured worker result'
    assert all(row['workerResult'] and row['workerResult']['executionPath'] == 'metal-local'
               and row['outcome'] == 'completed' and row['firstTokenReceived'] for row in observed)
    assert all(row['prefillSeconds'] is not None and
               (row['outputTokens'] < 2 or row['decodeSeconds'] is not None) for row in observed)
    for row in observed:
        timing = row['workerResult']['timings']
        stages = timing['prefillStages']
        assert all(stages[key] == 0 for key in ('cudaComputeMs', 'cudaElapsedMs', 'wirePayloadBytes', 'macStateImportMs'))
        assert timing['decodeTokens'] == max(row['outputTokens'] - 1, 0)
    assert sum(row['inputTokens'] for row in observed) == summary['tokens']['input']
    assert sum(row['outputTokens'] for row in observed) == summary['tokens']['output']
    health = read(base/'candidate-final-health.json')
    assert SHA in health['build'] and health['kv'] == 'fp32'
    package = read(base/'release/manifest.json')
    assert package['source'] == SHA
    assert all(sha256(base/'release'/name) == digest for name,digest in package['files'].items())
    assert functional['count'] == 20 and functional['negative_control'] == 'failed'
    assert all(row.get('exit_code') in (0, 1) or row['status'] in ('timeout', 'missing_solution') for row in functional['rows']), 'Container infrastructure failures cannot count as official assertion failures'
    checks = {row['task_id']: row['status'] for row in functional['rows']}
    data = read(ROOT/'site/trends.json')
    assert not any(row['id'] == RUN_ID for row in data['runs'])
    template = copy.deepcopy(data['configurations'][0])
    template.update({'id': CONFIG_ID, 'label': 'Issue #58 · M5 Pro only · Pi 20 · affine8 / FP32',
        'chunking': 'Existing full-model Metal prefill; managed chunk 32, scratch 128; fused FP32 decode; local threshold 262144',
        'topology': 'Single M5 Pro 48 GB, full model on Metal; Windows absent',
        'network': 'Loopback only; two streaming observation hops; no remote prefill or handoff',
        'power': 'AC / High Power; one model worker; serial cohort',
        'workload': 'Pi 1.4.0; seed 0; fixed 20; fresh session/workspace per task; timeout 300 s; context 32768; max output/request 4096; two separate 518-token warmup/reset requests',
        'plannedTasks': manifest['ids'], 'latestBaselineRun': None})
    template['timing'] = {'wall': 'Pi subprocess start to exit, including tools and files',
        'prefill': 'Sum worker prompt tokens / sum raw ServiceEvent.result.timings.prefillMs',
        'decode': 'Sum (output tokens minus one) / sum raw worker decode seconds',
        'ttft': 'client-send-to-first-valid-output-token'}
    requests, detail = [], []
    run_index = len(data['runs'])
    for task_index,task in enumerate(tasks):
        matched = [row for row in observed if instant(task['start_utc']) <= instant(row['completedUtc']) <= instant(task['end_utc'])]
        assert len(matched) == len(task['pi']['requests']), task['task_id']
        for field in ('input', 'output'):
            assert sum(row[field+'Tokens'] for row in matched) == task['pi']['usage'][field]
        official = checks[task['task_id']]
        status = 'passed' if official == 'passed' else 'failed' if official in ('failed','timeout') else 'error'
        detail.append({'taskId':task['task_id'], 'attempt':1, 'wallSeconds':task['wall_seconds'],
            'inputTokens':task['pi']['usage']['input'], 'outputTokens':task['pi']['usage']['output'],
            'modelRequests':len(matched), 'status':status,
            'officialValidation':'passed' if official=='passed' else 'failed' if official in ('failed','timeout') else 'not-run',
            'source': f'{SOURCE}#/runs/{run_index}/tasks/{task_index}'})
        for row in matched:
            index = len(requests)
            requests.append({key:row[key] for key in ('inputTokens','outputTokens','prefillSeconds','decodeSeconds','clientTtftMs','ttftBoundary','firstTokenReceived','outcome')})
            requests[-1].update({'id': f'issue58-local-{index+1:04d}', 'taskId':task['task_id'], 'attempt':1,
                'source': f'{SOURCE}#/runs/{run_index}/requests/{index}'})
    resources = lines(base/'resources.jsonl')
    final_sample = read(base/'final-resources.json')
    assert final_sample['candidatePidAlive'], 'Final resource boundary must precede model restoration'
    resources.append(final_sample)
    resources.sort(key=lambda sample: instant(sample['utc']))
    start,end = instant(tasks[0]['start_utc']), instant(tasks[-1]['end_utc'])
    before = [sample for sample in resources if instant(sample['utc']) <= start][-1]
    during = [sample for sample in resources if start <= instant(sample['utc']) <= end]
    assert during and all("'AC Power'" in sample['power'] for sample in during)
    after = [sample for sample in resources if instant(sample['utc']) >= end]
    last = after[0] if after else during[-1]
    resource_result = {'sampleIntervalSeconds':10, 'allObservedCohortPowerAc':True,
        'cohortStartUtc':tasks[0]['start_utc'], 'cohortEndUtc':tasks[-1]['end_utc'],
        'swapStartSampleUtc':before['utc'], 'swapEndSampleUtc':last['utc'],
        'swapBracketComplete': bool(after), 'sampledCohortSwapOutBytes':swapouts(last)-swapouts(before),
        'loadingAndWarmupSwapOutBytes':swapouts(before)-swapouts(resources[0]),
        'peakWorkerMetalAllocatedBytes':max(row['workerResult']['timings']['metalAllocatedBytes'] for row in observed)}
    input_tokens = sum(row['inputTokens'] for row in requests)
    output_tokens = sum(row['outputTokens'] for row in requests)
    prefill_s = sum(row['prefillSeconds'] for row in requests)
    decodable = [row for row in requests if row['outputTokens'] >= 2]
    intervals = sum(row['outputTokens']-1 for row in decodable)
    decode_s = sum(row['decodeSeconds'] for row in decodable)
    performance = {'runId':RUN_ID,'baselineSha':BASELINE,'macCommit':SHA,'windowsCommit':None,
        'metrics_complete':True,'piRequestCount':summary['request_count'],'workerMetricCount':len(requests),
        'inputTokens':input_tokens,'outputTokens':output_tokens,'prefillSeconds':prefill_s,
        'prefillTokensPerSecond':input_tokens/prefill_s,'decodeIntervals':intervals,'decodeSeconds':decode_s,
        'decodeTokensPerSecond':intervals/decode_s,'clientTtftMedianMs':statistics.median(row['clientTtftMs'] for row in requests),
        'piWallSeconds':summary['wall_seconds']['total'],'officialPassed':functional['passed'],
        'plannedTasks':20,'resources':resource_result}
    (base/'performance.json').write_text(json.dumps(performance,indent=2)+'\n')
    relative = lambda path: 'artifacts/'+base.name+'/'+str(path.relative_to(base))
    evidence_paths = [cohort/name for name in ('manifest.json','tasks.jsonl','summary.json')]
    evidence_paths += [base/name for name in ('cohort-observer/requests.jsonl','resources.jsonl','candidate-serve.log','warmup.json','functional.json','performance.json','release/manifest.json','observe.py','run.py','integration.log','restored-health.json','final-resources.json','environment.json','functional-mount-error.json','functional-mount-error.log')]
    evidence = {relative(path):{'sha256':sha256(path),'availability':'Local ignored artifacts; not a public download'} for path in evidence_paths}
    run = {'id':RUN_ID,'experimentId':'humaneval20-m5pro-20261024','configurationId':CONFIG_ID,
        'title':'Issue #58 · M5 Pro full local · HumanEval 20', 'date':'2026-10-24 America/Los_Angeles',
        'measuredCommit':SHA,'state':'pending-review','change':'Main eaaf81f plus explicit full-context Metal local routing; no inference kernel or precision changes.',
        'result':f"{summary['completion_count']}/20 Pi completed; {functional['passed']}/20 official assertions passed. Prefill {input_tokens/prefill_s:.2f} tok/s, decode {intervals/decode_s:.2f} tok/s; complete worker timings; all requests metal-local.",
        'provenance':{'kind':'candidate','mainCommit':None,'evidence':'https://github.com/LittleLittleCloud/Infer.NET/pull/59'},
        'baselineRunId':None,'runtime':{name:{'commit':SHA} for name in ('coordinator','macServe')},
        'conditions':{'windows':'Not participating','observer':'Two loopback hops; TTFT excludes Pi-to-observer dispatch','power':'AC / High Power','warmup':'Two deterministic 518-token structured requests, excluded','resources':resource_result},
        'binarySha256':{'macExecutable':package['files']['deepinfer'],'macMetalLibrary':package['files']['libdeepinfer_metal.dylib']},
        'binaryEvidence':REPORT,'harnessCommit':SHA,'harnessSha256':manifest['harness_sha256'],
        'requestCount':len(requests),'requests':requests,'tasks':detail,
        'reportedTotals':{'wallSeconds':summary['wall_seconds']['total'],'inputTokens':input_tokens,'outputTokens':output_tokens,'modelRequests':len(requests)},
        'rawEvidence':evidence,'links':[
            {'role':'issue','label':'Experiment Issue #58','url':'https://github.com/LittleLittleCloud/Infer.NET/issues/58'},
            {'role':'code-pr','label':'Local route code PR #59','url':'https://github.com/LittleLittleCloud/Infer.NET/pull/59'},
            {'role':'results-pr','label':'Results PR','url':args.results_pr},
            {'role':'report','label':'M5 Pro 20-task report','url':'https://github.com/LittleLittleCloud/Infer.NET/blob/main/'+REPORT}]}
    data['configurations'].append(template);data['runs'].append(run)
    committed = subprocess.check_output(['git','show','-s','--format=%cI',SHA],text=True).strip()
    data['commitTimeline'].append({'sha':SHA,'committedAt':committed.replace('+00:00','Z'), 'evidence':'https://github.com/LittleLittleCloud/Infer.NET/commit/'+SHA})
    # git uses the local commit offset; normalize it explicitly for the timeline.
    from datetime import timezone
    data['commitTimeline'][-1]['committedAt'] = instant(committed).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    normalized = copy.deepcopy(data)
    normalized['runs'] = [import_pi(spec, ROOT) for spec in data['imports']] + normalized['runs']
    validate(normalized)
    (ROOT/'site/trends.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    public = ROOT/'docs/evaluations/issue58'
    public.mkdir(parents=True,exist_ok=True)
    (public/'performance.json').write_text(json.dumps(performance,ensure_ascii=False,indent=2)+'\n')
    (public/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(performance,indent=2))


if __name__ == '__main__':
    main()
