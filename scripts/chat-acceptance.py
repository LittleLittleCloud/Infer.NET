# coding: utf-8
"""Validate real deployed text chat with official openai/anthropic Python SDKs (no GPU simulation)."""
import argparse,json,pathlib,time
from openai import OpenAI
from anthropic import Anthropic
p=argparse.ArgumentParser();p.add_argument('--endpoint',default='http://127.0.0.1:5200');p.add_argument('--token-file',required=True);p.add_argument('--output',required=True);args=p.parse_args()
root=pathlib.Path(args.output);root.mkdir(parents=True,exist_ok=False)
token=pathlib.Path(args.token_file).read_text().strip()
o=OpenAI(base_url=args.endpoint.rstrip('/')+'/v1',api_key=token,timeout=320,max_retries=0)
a=Anthropic(base_url=args.endpoint.rstrip('/'),api_key=token,timeout=320,max_retries=0)
model='qwen3.6-35b-a3b'
system='Answer only the requested color, without explanation.'
history=[{'role':'user','content':'Remember the color blue.'},{'role':'assistant','content':'I will remember blue.'},{'role':'user','content':'Which color did I ask you to remember?'}]
checks=[];observations=[]
def check(name,condition):
 checks.append({'name':name,'pass':bool(condition)});print(('PASS ' if condition else 'FAIL ')+name,flush=True)
 (root/'summary.json').write_text(json.dumps({'checks':checks,'observations':observations},ensure_ascii=False,indent=2))
 if not condition:raise AssertionError(name)
def save(name,result):
 (root/(name+'.json')).write_text(result.model_dump_json(indent=2))
def timed(name,call):
 start=time.monotonic(); result=call(); observations.append({'name':name,'elapsedSeconds':time.monotonic()-start}); save(name,result);return result
messages=[{'role':'system','content':system}]+history
full=timed('chat-full',lambda:o.chat.completions.create(model=model,messages=messages,max_tokens=32))
text=full.choices[0].message.content
check('Chat full remembers prior user/assistant turns','blue' in text.lower() and full.choices[0].finish_reason=='stop')
start=time.monotonic();arr=[];pieces=[];times=[]
with o.chat.completions.create(model=model,messages=messages,max_tokens=32,stream=True,stream_options={'include_usage':True}) as stream:
 for c in stream:
  arr.append(c.model_dump())
  if c.choices and c.choices[0].delta.content:pieces.append(c.choices[0].delta.content);times.append(time.monotonic()-start)
(root/'chat-stream.json').write_text(json.dumps(arr,ensure_ascii=False,indent=2));observations.append({'name':'chat-stream','textEventTimesSeconds':times,'elapsedSeconds':time.monotonic()-start})
check('Chat stream text and exact token usage agree',''.join(pieces)==text and arr[-1]['usage']==full.usage.model_dump())
m=timed('messages-full',lambda:a.messages.create(model=model,system=system,messages=history,max_tokens=32))
check('Messages full agrees with Chat',m.content[0].text==text and m.stop_reason=='end_turn' and m.usage.input_tokens==full.usage.prompt_tokens and m.usage.output_tokens==full.usage.completion_tokens)
start=time.monotonic();parts=[];times=[]
with a.messages.stream(model=model,system=system,messages=history,max_tokens=32) as stream:
 for v in stream.text_stream:parts.append(v);times.append(time.monotonic()-start)
 final=stream.get_final_message()
save('messages-stream-final',final);observations.append({'name':'messages-stream','textEventTimesSeconds':times,'elapsedSeconds':time.monotonic()-start})
check('Messages SDK stream accumulates exact text and usage',''.join(parts)==text and final.content[0].text==text and final.usage==m.usage)
r=timed('responses-full',lambda:o.responses.create(model=model,instructions=system,input=history,max_output_tokens=32,store=False))
check('Responses full agrees with Chat',r.output_text==text and r.status=='completed' and r.usage.input_tokens==full.usage.prompt_tokens and r.usage.output_tokens==full.usage.completion_tokens)
roundtrip=timed('responses-history',lambda:o.responses.create(model=model,instructions=system,input=history+[item.model_dump(exclude_none=True) for item in r.output]+[{'role':'user','content':'Repeat the same color.'}],max_output_tokens=32,store=False))
check('Responses output history round trip works with real model','blue' in roundtrip.output_text.lower())
start=time.monotonic();parts=[];times=[]
with o.responses.stream(model=model,instructions=system,input=history,max_output_tokens=32,store=False) as stream:
 for event in stream:
  if event.type=='response.output_text.delta':parts.append(event.delta);times.append(time.monotonic()-start)
 final=stream.get_final_response()
save('responses-stream-final',final);observations.append({'name':'responses-stream','textEventTimesSeconds':times,'elapsedSeconds':time.monotonic()-start})
check('Responses SDK stream accumulates exact text and usage',''.join(parts)==text and final.output_text==text and final.usage==r.usage)
long=[{'role':'user','content':'Write a detailed story about a cat visiting a library.'}]
start=time.monotonic();parts=[];times=[]
with o.chat.completions.create(model=model,messages=long,max_tokens=64,stream=True) as stream:
 for c in stream:
  if c.choices and c.choices[0].delta.content:parts.append(c.choices[0].delta.content);times.append(time.monotonic()-start)
observations.append({'name':'long-stream','text':''.join(parts),'textEventTimesSeconds':times,'elapsedSeconds':time.monotonic()-start})
check('Real deltas arrive before final completion',len(times)>5 and times[-1]-times[0]>.1)
with o.chat.completions.create(model=model,messages=long,max_tokens=128,stream=True) as stream:
 for c in stream:
  if c.choices and c.choices[0].delta.content:break
check('Client closes real chat stream after first text',True)
time.sleep(1)
after=timed('after-cancel',lambda:o.chat.completions.create(model=model,messages=messages,max_tokens=32))
check('Next chat correct after cancellation',after.choices[0].message.content==text and after.usage==full.usage)
r=timed('responses-length',lambda:o.responses.create(model=model,input=long,max_output_tokens=8,store=False))
check('Response output limit reports incomplete',r.status=='incomplete' and r.incomplete_details.reason=='max_output_tokens' and r.usage.output_tokens==8)
check('Models endpoint advertises real chat model',any(m.id==model for m in o.models.list().data))
for i in range(3):
 result=timed('short-turn-'+str(i),lambda:o.chat.completions.create(model=model,messages=messages,max_tokens=32))
 check('Immediate next short turn '+str(i),result.choices[0].message.content==text)
print('Completed',len(checks),'real chat checks',flush=True)
