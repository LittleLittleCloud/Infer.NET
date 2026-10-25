#!/usr/bin/env python3
"""Import audited Issue72 records without changing the adopted baseline selection."""
import copy,json,pathlib,subprocess,sys
from datetime import datetime,timezone
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from home_trends import validate,import_pi
REPO='https://github.com/LittleLittleCloud/Infer.NET';PUBLIC=REPO+'/blob/main/'
REPORT='docs/EXPERIMENT-72-SPECULATIVE-STATE-20261024.md'
data=json.loads((ROOT/'site/trends.json').read_text())
for mode in ['baseline','off','on']:
 path=ROOT/'docs/evaluations/issue72'/(mode+'.json')
 if not path.exists():continue
 m=json.loads(path.read_text());rid=m['runId'];diagnostic=mode=='on';sha=m['macCommit']
 assert m['validForDeclaredComparison'] and m['attemptedTasks']==m['plannedTasks']==20
 existing=next((i for i,r in enumerate(data['runs']) if r['id']==rid),None)
 cfg_id='issue72-synthetic-correctness-human20' if diagnostic else 'issue72-local-fp32-human20'
 if not any(c['id']==cfg_id for c in data['configurations']):
  cfg=copy.deepcopy(data['configurations'][0]);cfg.update({'id':cfg_id,'label':'Issue72 · synthetic oracle correctness only' if diagnostic else 'Issue72 · main / candidate-off comparison','model':'mlx-community/Qwen3.6-35B-A3B-8bit','modelRevision':'e06a74e6236a60c8367e1a3214e83d8b61b637b0','weights':'affine8/group64; original per-matrix formats','kv':'FP32','cache':'0% prefix hits; request state reset','chunking':'Local prefill32 / scratch128 / fused-fp32; synthetic mixed B4' if diagnostic else 'Local prefill32 / scratch128 / fused-fp32; no speculation','topology':'Single M5 Pro20 GPU/48GiB, metal-local, Windows absent','network':'Two loopback observation hops; no handoff','power':'AC / High Power; default wired limit; serial requests','workload':'Pi1.4.0, fixed20/seed0, context32768/output4096/task timeout300s; fresh sessions; preflight warmup excluded','datasetRevision':'b9453e8f1865df0a0057a1383767caa400acfbcf','datasetSha256':m['datasetSha256'],'plannedTasks':[t['taskId'] for t in m['tasks']],'latestBaselineRun':None})
  cfg['timing']={'wall':'Pi process start to exit including tools','prefill':'Standard speed not evaluated for synthetic-oracle correctness mode; measured worker prefill resources remain in the public JSON' if diagnostic else 'Sum worker inputs / sum worker prefill seconds','decode':'N-1 generation-equivalent clock unavailable; separate cycle/emission diagnostics' if diagnostic else 'Sum output-minus-one intervals / sum worker decode seconds','ttft':'client-send-to-first-valid-output-token'}
  data['configurations'].append(cfg)
 if not any(t['sha']==sha for t in data['commitTimeline']):
  data['commitTimeline'].append({'sha':sha,'committedAt':datetime.fromisoformat(subprocess.check_output(['git','show','-s','--format=%cI',sha],cwd=ROOT,text=True).strip()).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),'evidence':REPO+'/commit/'+sha})
 source=PUBLIC+str(path.relative_to(ROOT));requests=[];tasks=[]
 for i,r in enumerate(m['requests']):
  q={k:r[k] for k in ['taskId','attempt','inputTokens','outputTokens','prefillSeconds','decodeSeconds','clientTtftMs','ttftBoundary','firstTokenReceived','outcome']};q.update(id=rid+'-'+str(i+1),source=source+'#/requests/'+str(i))
  if diagnostic:q.update(prefillSeconds=None,decodeSeconds=None)
  requests.append(q)
 for i,t in enumerate(m['tasks']):
  t=copy.deepcopy(t);t['source']=source+'#/tasks/'+str(i);obs=[r for r in requests if r['taskId']==t['taskId']]
  for field in ['inputTokens','outputTokens']:
   if obs and all(r[field] is not None for r in obs):t[field]=sum(r[field] for r in obs)
  tasks.append(t)
 speed='Synthetic correctness mode: no standard speed point; measured resources/cycles in public JSON.' if diagnostic else f"Worker prefill {m['prefillTokensPerSecond']:.3f}, decode {m['decodeTokensPerSecond']:.3f} tok/s."
 run={'id':rid,'experimentId':'speculative-state-20261024','configurationId':cfg_id,'title':'Issue72 · '+mode+' · HumanEval20','date':'2026-10-24 America/Los_Angeles','measuredCommit':sha,'state':'failed' if m['officialPassed']!=20 or m['piCompleted']!=20 else 'pending-review','change':'Fixed main reference.' if mode=='baseline' else 'Default-off state transaction candidate, synthetic oracle '+('enabled.' if diagnostic else 'disabled.'),'result':f"Pi {m['piCompleted']}/20; official {m['officialPassed']}/20. "+speed,'provenance':{'kind':'main','mainCommit':sha,'evidence':REPO+'/commit/'+sha} if mode=='baseline' else {'kind':'candidate','evidence':REPO+'/pull/74'},'baselineRunId':'issue72-main-human20-20261024' if mode=='off' else None,'runtime':{k:{'commit':sha} for k in ['coordinator','macServe']},'conditions':{'adoption':'Human review pending; no baseline selection, deployment or adoption change.','observer':'TTFT from observer send to first valid output; Pi-to-observer dispatch excluded; two forwards.','resources':m['resources'],'piUsageMatchesWorker':m['piUsageMatchesWorker'],'mode':'Synthetic oracle correctness only; no real draft or acceleration claim.' if diagnostic else 'Clean worker timing cohort.'},'binarySha256':m['binarySha256'],'binaryEvidence':REPORT,'harnessCommit':sha,'harnessSha256':m['harnessSha256'],'requestCount':len(requests),'requests':requests,'tasks':tasks,'reportedTotals':{'wallSeconds':m['piWallSeconds'],'inputTokens':sum(t['inputTokens'] or 0 for t in tasks),'outputTokens':sum(t['outputTokens'] or 0 for t in tasks),'modelRequests':len(requests)},'rawEvidence':m['localEvidence'],'links':[{'role':'issue','label':'Issue72','url':REPO+'/issues/72'},{'role':'code-pr','label':'Code PR74','url':REPO+'/pull/74'},{'role':'results-pr','label':'Results PR75','url':REPO+'/pull/75'},{'role':'report','label':'Full record','url':PUBLIC+REPORT},{'role':'evidence','label':'Audited JSON','url':source},{'role':'notebook','label':'Notebook','url':'experiments.html#speculative-state-issue72-20261024'}]}
 if existing is None:data['runs'].append(run)
 else:data['runs'][existing]=run
normalized=copy.deepcopy(data)
normalized['runs']=[import_pi(spec,ROOT) for spec in data['imports']]+normalized['runs']
validate(normalized)
(ROOT/'site/trends.json').write_text(json.dumps(data,indent=2)+'\n')
print('Imported audited Issue72 records; baseline selection unchanged.')
