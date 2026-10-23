#!/usr/bin/env python3
"""Protocol tests using a controlled text worker, no GPU or cloud credentials required."""
import argparse,concurrent.futures,http.server,json,os,pathlib,socket,subprocess,threading,time,urllib.request,urllib.error
p=argparse.ArgumentParser();p.add_argument('--executable');p.add_argument('--sdk',action='store_true');p.add_argument('--tui',action='store_true');a=p.parse_args();root=pathlib.Path(__file__).resolve().parents[1]
def port():
 with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
(root/'artifacts/chat').mkdir(parents=True,exist_ok=True)
cp,wp=port(),port();token='chat-test-token-not-production-12345';seen=[]
class Worker(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def handle(self):
  try:super().handle()
  except (BrokenPipeError,ConnectionResetError):pass
 def do_POST(self):
  if self.headers.get('Transfer-Encoding','').lower()=='chunked':
   raw=b''
   while True:
    size=int(self.rfile.readline().strip(),16)
    if size==0:self.rfile.readline();break
    raw+=self.rfile.read(size);self.rfile.read(2)
  else:raw=self.rfile.read(int(self.headers['Content-Length']))
  q=json.loads(raw);seen.append(q)
  assert self.headers['X-DeepInfer-Token']==token
  text='你好 [world]';r=dict(requestId='test',workerId='test',model=q['model'],backend='metal',text=text,tokens=[],isSimulation=True,computeMs=1,tokenIds=[1,2],promptTokens=23,finishReason='eos')
  last=q['messages'][-1]['content']
  if last=='busy':self.send_response(429);self.end_headers();self.wfile.write(b'{"error":"busy"}');return
  self.send_response(200);self.send_header('Content-Type','text/event-stream' if self.path.endswith('/stream') else 'application/json');self.end_headers()
  if not self.path.endswith('/stream'):self.wfile.write(json.dumps(r).encode());return
  for text in ['你','好 [world]']:
   v=dict(text=text,tokenId=1,result=None,error=None,promptTokens=23)
   self.wfile.write(('event: token\ndata: '+json.dumps(v)+'\n\n').encode());self.wfile.flush()
   if last=='slow':time.sleep(2)
  if last=='truncate':return
  if last=='fault':v=dict(error={'code':'failed','message':'fault during stream'})
  else:v=dict(text=None,tokenId=None,result=r,error=None)
  self.wfile.write(('event: done\ndata: '+json.dumps(v)+'\n\n').encode());self.wfile.flush()
server=http.server.ThreadingHTTPServer(('127.0.0.1',wp),Worker);threading.Thread(target=server.serve_forever,daemon=True).start()
cmd=[a.executable] if a.executable else [os.environ.get('DOTNET','dotnet'),str(root/'src/DeepInfer.Host/bin/Release/net10.0/deepinfer.dll')]
process=subprocess.Popen(cmd+['coordinator','--listen',f'http://127.0.0.1:{cp}','--token',token],stdout=open(root/'artifacts/chat/protocol-host.log','w'),stderr=subprocess.STDOUT)
def call(path,body=None,headers=None):
 req=urllib.request.Request(f'http://127.0.0.1:{cp}'+path,data=None if body is None else json.dumps(body).encode(),headers=headers or {'Authorization':'Bearer '+token,'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=20) as r:return r.status,r.read().decode(),r.headers
 except urllib.error.HTTPError as e:return e.code,e.read().decode(),e.headers
checks=0
def check(ok,label):
 global checks
 assert ok,label;checks+=1;print('PASS',label,flush=True)
try:
 for _ in range(100):
  try:
   if call('/health')[0]==200:break
  except OSError:pass
  time.sleep(.1)
 reg=dict(protocolVersion=1,workerId='test',endpoint=f'http://127.0.0.1:{wp}',devices=[dict(id='metal',backend='metal',name='test',architecture='test',memoryBytes=None,nativeAvailable=True,supportsInference=True,status='test')],models=[dict(id='qwen3.6-35b-a3b',format='test-chat-v1',isSimulation=True)],capacity=1)
 check(call('/v1/workers/register',reg,{'X-DeepInfer-Token':token,'Content-Type':'application/json'})[0]==204,'register controlled worker')
 check(call('/v1/models')[0]==200,'Bearer auth/models')
 for path in ['/v1/chat/completions','/v1/messages','/v1/responses']:
  ismsg=path.endswith('messages');isresp=path.endswith('responses')
  body={'model':'qwen3.6-35b-a3b',('input' if isresp else 'messages'):[{'role':'user','content':'hi'}],('max_output_tokens' if isresp else 'max_tokens'):32}
  headers={'x-api-key':token,'anthropic-version':'2023-06-01','Content-Type':'application/json'} if ismsg else None
  code,s,_=call(path,body,headers);v=json.loads(s);check(code==200,'nonstream '+path)
  check((v['output'][0]['content'][0]['text'] if isresp else v['content'][0]['text'] if ismsg else v['choices'][0]['message']['content'])=='你好 [world]','text '+path)
  check(seen[-1]['messages']==[{'role':'user','content':'hi'}],'structured history '+path)
  body['stream']=True
  if not ismsg and not isresp:body['stream_options']={'include_usage':True}
  code,s,h=call(path,body,headers);events=[json.loads(x[6:]) for x in s.splitlines() if x.startswith('data: ') and x!='data: [DONE]']
  check(code==200 and 'text/event-stream' in h['Content-Type'],'SSE '+path)
  if ismsg:
   check([x['type'] for x in events]==['message_start','content_block_start','content_block_delta','content_block_delta','content_block_stop','message_delta','message_stop'],'Messages event lifecycle')
   check(events[0]['message']['usage']['input_tokens']==23 and events[-2]['usage']['output_tokens']==2,'Messages usage')
  elif isresp:
   check(events[0]['type']=='response.created' and events[-1]['type']=='response.completed','Responses lifecycle')
   check([x['sequence_number'] for x in events]==list(range(len(events))),'Responses sequence')
   check(events[-1]['response']['output'][0]['content'][0]['text']=='你好 [world]','Responses final output')
  else:
   check(s.endswith('data: [DONE]\n\n') and events[-1]['usage']['total_tokens']==25,'Chat completion terminal usage')
  body['input' if isresp else 'messages']=[{'role':'user','content':'busy'}]
  code,s,_=call(path,body,headers);check(code==429 and 'error' in json.loads(s),'prestream backpressure '+path)
  body['input' if isresp else 'messages']=[{'role':'user','content':'truncate'}]
  code,s,_=call(path,body,headers);check('partial output' in s and '[DONE]' not in s and 'response.completed' not in s and 'message_stop' not in s,'truncated upstream '+path)
  body['tools']=[];check(call(path,body,headers)[0]==400,'unsupported options '+path)
  check(call(path,body,{'Authorization':'Bearer wrong','Content-Type':'application/json'})[0]==401,'auth rejected '+path)
 if a.sdk:
  from openai import OpenAI
  from anthropic import Anthropic
  with OpenAI(base_url=f'http://127.0.0.1:{cp}/v1',api_key=token) as sdk:
   args=dict(model='qwen3.6-35b-a3b',messages=[dict(role='user',content='hi')],max_tokens=32)
   check(sdk.chat.completions.create(**args).choices[0].message.content=='你好 [world]','OpenAI SDK complete')
   stream=list(sdk.chat.completions.create(**args,stream=True,stream_options={'include_usage':True}))
   check(''.join(x.choices[0].delta.content or '' for x in stream if x.choices)=='你好 [world]' and stream[-1].usage.total_tokens==25,'OpenAI SDK chat stream')
   check(sdk.responses.create(model='qwen3.6-35b-a3b',input='hi',max_output_tokens=32,store=False).output_text=='你好 [world]','OpenAI SDK Responses complete')
   with sdk.responses.stream(model='qwen3.6-35b-a3b',input='hi',max_output_tokens=32,store=False) as stream:
    for event in stream:pass
    check(stream.get_final_response().output_text=='你好 [world]','OpenAI SDK Responses stream accumulator')
  with Anthropic(base_url=f'http://127.0.0.1:{cp}',api_key=token) as sdk:
   check(sdk.messages.create(**args).content[0].text=='你好 [world]','Anthropic SDK complete')
   with sdk.messages.stream(**args) as stream:
    text=''.join(stream.text_stream);final=stream.get_final_message()
    check(text=='你好 [world]' and final.usage.input_tokens==23 and final.usage.output_tokens==2,'Anthropic SDK stream accumulator')
 if a.tui:
  import pexpect,pyte
  # Refresh the short worker lease throughout terminal interaction.
  keep=threading.Event()
  def heartbeats():
   while not keep.wait(2):call('/v1/workers/register',reg,{'X-DeepInfer-Token':token,'Content-Type':'application/json'})
  threading.Thread(target=heartbeats,daemon=True).start()
  env=dict(os.environ,DEEPINFER_TOKEN=token,TERM='xterm-256color');env.pop('DEEPINFER_TOKEN_FILE',None)
  child=pexpect.spawn(os.environ.get('DOTNET','dotnet'),[str(root/'src/DeepInfer.Tui/bin/Release/net10.0/deepinfer-tui.dll'),f'http://127.0.0.1:{cp}'],env=env,encoding='utf-8',dimensions=(30,100),timeout=1)
  screen=pyte.Screen(100,30);terminal=pyte.Stream(screen);raw=[]
  def pump(seconds):
   end=time.monotonic()+seconds
   while time.monotonic()<end:
    try:
     data=child.read_nonblocking(65536,timeout=.2);raw.append(data)
     if '\x1b[6n' in data:child.send('\x1b[1;1R')
     terminal.feed(data)
    except pexpect.TIMEOUT:pass
  try:
   pump(3);before='\n'.join(screen.display);(root/'artifacts/chat/tui-before.txt').write_text(before)
   check('Chat playground' in before and 'Ready' in before,'TUI connected')
   child.send('\r');pump(3);after='\n'.join(screen.display);(root/'artifacts/chat/tui-after.txt').write_text(after)
   check('Done' in after and 'world' in after,'TUI Enter generates streaming response through IChatClient')
   check('[world]' in after and '[[world]]' not in after,'TUI preserves literal brackets')
   check(not any(c in before+after for c in '┌┐└┘│─╭╮╰╯'),'TUI has no borders')
   child.send('remember?\r');pump(3)
   (root/'artifacts/chat/tui-history.json').write_text(json.dumps(seen[-1],ensure_ascii=False));(root/'artifacts/chat/tui-second.txt').write_text('\n'.join(screen.display))
   check(len(seen[-1]['messages'])==4 and seen[-1]['messages'][-2]['role']=='assistant' and seen[-1]['messages'][-1]['content']=='remember?','TUI sends multi-turn history')
   child.send('\t\t\t\r');pump(.5)
   check('New conversation.' in '\n'.join(screen.display),'TUI new chat clears transcript')
   child.send('\x1b[Z'*3);child.send('slow\r');pump(.8)
   check('Streaming' in '\n'.join(screen.display),'TUI renders before completion')
   child.send('\t\t\r');pump(.6)
   check('Stopped' in '\n'.join(screen.display),'TUI Stop cancels generation')
   child.send('\x1b[Z'*2);child.send('resume\r');pump(3)
   check(len(seen[-1]['messages'])==2 and seen[-1]['messages'][-1]['content']=='resume','TUI excludes cancelled history')
  finally:
   keep.set();child.sendcontrol('c');child.close(force=True);(root/'artifacts/chat/tui-session.ansi').write_text(''.join(raw))
 print('Completed',checks,'chat protocol checks')
finally:process.terminate();process.wait(timeout=10);server.shutdown()
