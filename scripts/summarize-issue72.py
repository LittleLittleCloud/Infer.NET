#!/usr/bin/env python3
"""Audit real Issue72 cohorts without exporting prompts, model text or solutions."""
import argparse,datetime,hashlib,json,math,pathlib,re,statistics

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def instant(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def swap(s):
 return int(re.search(r'^Swapouts:\s+(\d+)',s,re.M)[1])*int(re.search(r'page size of (\d+) bytes',s)[1])
def finite(v):return type(v) in (int,float) and math.isfinite(v) and v>=0

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',type=pathlib.Path,required=True);p.add_argument('--observations',type=pathlib.Path,required=True)
 p.add_argument('--skip-rows',type=int,default=0);p.add_argument('--end-row',type=int)
 p.add_argument('--functional',type=pathlib.Path);p.add_argument('--package',type=pathlib.Path,required=True)
 p.add_argument('--mode',choices=['baseline','off','on','pilot'],required=True)
 p.add_argument('--run-id',required=True);p.add_argument('--valid-for-comparison',action='store_true')
 p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 manifest=read(a.run/'manifest.json');summary=read(a.run/'summary.json');tasks=lines(a.run/'tasks.jsonl')
 assert manifest['seed']==0 and manifest['pi_version']=='1.4.0'
 assert manifest['dataset_revision']=='b9453e8f1865df0a0057a1383767caa400acfbcf'
 assert manifest['dataset_sha256']=='b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef'
 assert manifest['model_revision']=='e06a74e6236a60c8367e1a3214e83d8b61b637b0'
 assert manifest['context_window']==32768 and manifest['max_tokens']==4096 and manifest['timeout_seconds']==300
 assert [t['task_id'] for t in tasks]==manifest['ids'][:len(tasks)]
 if a.valid_for_comparison:assert len(tasks)==len(manifest['ids']), 'A comparable cohort must retain every planned task attempt'
 package=read(a.package/'manifest.json');assert package['source']==manifest['endpoint_build']
 assert all(digest(a.package/name)==sha for name,sha in package['files'].items())
 observations=lines(a.observations)[a.skip_rows:a.end_row]
 # Explicit UTC interval excludes preflight/warmup and retains all in-run failures.
 start=instant(tasks[0]['start_utc']) if tasks else None;end=instant(tasks[-1]['end_utc']) if tasks else None
 rows=[x for x in observations if start and start<=instant(x['startedUtc'])<=end]
 official=read(a.functional) if a.functional else None
 if official:
  assert official['count']==len(manifest['ids'])==20 and official['negative_control']=='failed'
  assert all(x.get('exit_code') in (0,1) or x['status'] in ['timeout','missing_solution'] for x in official['rows'])
 checks={x['task_id']:x['status'] for x in official['rows']} if official else {}
 requests=[];public_tasks=[]
 for task_id in manifest['ids']:
  t=next((t for t in tasks if t['task_id']==task_id),None)
  matched=[x for x in rows if t and instant(t['start_utc'])<=instant(x['startedUtc'])<=instant(t['end_utc'])]
  status=checks.get(task_id)
  outcome='passed' if status=='passed' else 'failed' if status in ['failed','timeout'] else 'timeout' if t and t.get('timeout') else 'error' if status=='missing_solution' or t and t.get('error') else 'unverified' if t else 'incomplete'
  usage=t['pi']['usage'] if t else {}
  public_tasks.append({'taskId':task_id,'attempt':1,'status':outcome,'officialValidation':'passed' if status=='passed' else 'failed' if status in ['failed','timeout'] else 'unverified' if t and not t.get('error') and not t.get('timeout') and status!='missing_solution' else 'not-run','wallSeconds':t['wall_seconds'] if t else None,'inputTokens':usage.get('input'),'outputTokens':usage.get('output'),'modelRequests':len(matched) if t else None})
  for x in matched:
   worker=x.get('workerResult') or {};timing=worker.get('timings') or {};diagnostic=(worker.get('diagnostics') or {}).get('speculative') or {};resources=diagnostic.get('resources') or {}
   prefill=x.get('prefillSeconds');source='worker-timings'
   if prefill is None and finite(resources.get('prefillMs')):prefill=resources['prefillMs']/1000;source='speculative-resources-worker-prefill'
   decode=x.get('decodeSeconds')
   if a.mode=='on':assert not timing and decode is None
   if worker:assert worker['executionPath']=='metal-local'
   before=x.get('resourcesBefore');after=x.get('resourcesAfter');added=swap(after['vmStat'])-swap(before['vmStat']) if before and after else None
   req={'id':x['id'],'taskId':task_id,'attempt':1,'inputTokens':x.get('inputTokens'),'outputTokens':x.get('outputTokens'),'prefillSeconds':prefill,'decodeSeconds':decode,'clientTtftMs':x.get('clientTtftMs'),'ttftBoundary':x['ttftBoundary'],'firstTokenReceived':x['firstTokenReceived'],'outcome':x['outcome'],'prefillClock':source,'metalAllocatedBytes':timing.get('metalAllocatedBytes',resources.get('metalAllocatedBytes')),'contextCapacity':timing.get('contextCapacity',resources.get('contextCapacity')),'snapshotHostBytes':resources.get('snapshotHostBytes'),'systemSwapOutBytes':added,'powerAc':all("'AC Power'" in z['power'] for z in [before,after]) if before and after else None}
   if diagnostic:req['speculative']={k:v for k,v in diagnostic.items() if k not in ['detail','note','resources']}
   requests.append(req)
 assert len(requests)==len(rows),'Requests outside task windows or duplicate matching'
 pre=[x for x in requests if x['inputTokens'] is not None and finite(x['prefillSeconds']) and x['prefillSeconds']>0]
 dec=[x for x in requests if x['outputTokens'] is not None and x['outputTokens']>=2 and finite(x['decodeSeconds']) and x['decodeSeconds']>0]
 tt=[x['clientTtftMs'] for x in requests if finite(x['clientTtftMs'])]
 inputs=sum(x['inputTokens'] for x in requests if x['inputTokens'] is not None);outputs=sum(x['outputTokens'] for x in requests if x['outputTokens'] is not None)
 pre_s=sum(x['prefillSeconds'] for x in pre);dec_s=sum(x['decodeSeconds'] for x in dec);intervals=sum(x['outputTokens']-1 for x in dec)
 swaps=[x['systemSwapOutBytes'] for x in requests if x['systemSwapOutBytes'] is not None];metals=[x['metalAllocatedBytes'] for x in requests if x['metalAllocatedBytes'] is not None]
 result={'schema':'issue72-cohort-v1','runId':a.run_id,'mode':a.mode,'validForPerformanceComparison':a.valid_for_comparison,'macCommit':manifest['endpoint_build'],'windowsCommit':None,'startedUtc':tasks[0]['start_utc'] if tasks else None,'endedUtc':tasks[-1]['end_utc'] if tasks else None,'plannedTasks':len(manifest['ids']),'attemptedTasks':len(tasks),'piCompleted':summary['completion_count'],'piErrors':summary['error_count'],'piTimeouts':summary['timeout_count'],'officialPassed':official['passed'] if official else None,'piRequestCount':summary['request_count'],'observedRequestCount':len(requests),'workerInputTokens':inputs,'workerOutputTokens':outputs,'piUsageMatchesWorker':inputs==summary['tokens']['input'] and outputs==summary['tokens']['output'],'prefillSeconds':pre_s if pre else None,'prefillTokensPerSecond':sum(x['inputTokens'] for x in pre)/pre_s if pre_s else None,'decodeIntervals':intervals if dec else None,'decodeSeconds':dec_s if dec else None,'decodeTokensPerSecond':intervals/dec_s if dec_s else None,'clientTtftMedianMs':statistics.median(tt) if tt else None,'coverage':{'prefillRequests':len(pre),'decodeRequests':len(dec),'clientTtftRequests':len(tt),'totalObservedRequests':len(requests)},'piWallSeconds':summary['wall_seconds']['total'],'resources':{'maxObservedMetalAllocatedBytes':max(metals) if metals else None,'metalObservation':'Response samples, not a continuous allocator high-water trace','maxPerRequestSystemSwapOutBytes':max(swaps) if swaps else None,'sumPerRequestSystemSwapOutBytes':sum(swaps) if swaps else None,'allObservedRequestPowerAc':all(x['powerAc'] is True for x in requests),'requestResourceCoverage':len(swaps),'perRequestSwapLimitBytes':512*2**20,'metalAllocationLimitBytes':int(38.5*2**30)},'binarySha256':{'macExecutable':package['files']['deepinfer'],'macMetalLibrary':package['files']['libdeepinfer_metal.dylib']},'harnessSha256':manifest['harness_sha256'],'datasetSha256':manifest['dataset_sha256'],'tasks':public_tasks,'requests':requests,'localEvidence':{name:{'sha256':digest(path),'availability':'Local ignored experiment artifacts; not a public download'} for name,path in [('manifest.json',a.run/'manifest.json'),('summary.json',a.run/'summary.json'),('tasks.jsonl',a.run/'tasks.jsonl'),('observations.jsonl',a.observations)]}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['tasks','requests','localEvidence']},indent=2))
if __name__=='__main__':main()
