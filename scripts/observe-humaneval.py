# Reproduction copy: only machine-specific configuration lines were parameterized.
# SHA-256 of the original executed local script is preserved with raw evidence.
"""Temporary loopback observation for evaluation; never saves request/response text."""
import argparse
import datetime as dt
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request
from urllib.parse import urlsplit

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config-dir',type=Path,required=True,help='Existing private Mac config directory; never logged')
parser.add_argument('--executable',type=Path,required=True,help='Verified same-main Native AOT coordinator executable')
parser.add_argument('--outer-port',type=int,default=5280)
parser.add_argument('--clone-port',type=int,default=5281)
parser.add_argument('--worker-port',type=int,default=5282)
parser.add_argument('--output',type=Path,required=True,help='Dedicated observer output directory')
args=parser.parse_args()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
PRIVATE=args.config_dir.resolve()
CONFIG=json.loads((PRIVATE/'coordinator.json').read_text())
TOKEN=Path(CONFIG['token-file']).read_text().strip()
SOURCE=urlsplit(CONFIG['listen'])
CLONE=f'http://127.0.0.1:{args.clone_port}'
WORKER=json.loads((PRIVATE/'worker.json').read_text())['listen']
EXE=args.executable.resolve()
stop=threading.Event(); mutex=threading.Lock(); gate=threading.Lock()
active=None; sequence=sum(1 for _ in (OUT/'requests.jsonl').open()) if (OUT/'requests.jsonl').exists() else 0

def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()
def call(url,body=None):
    req=urllib.request.Request(url,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json','X-DeepInfer-Token':TOKEN})
    with urllib.request.urlopen(req,timeout=5) as r:
        value=r.read()
        return json.loads(value) if value else None

class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*args): pass
    def do_GET(self): self.forward()
    def do_POST(self): self.forward()
    def forward(self):
        global active,sequence
        is_outer=self.server.server_port==args.outer_port
        measured=is_outer and self.command=='POST' and self.path.split('?')[0] in ['/v1/chat/completions','/v1/messages','/v1/responses']
        if measured: gate.acquire()
        row=None; conn=None
        try:
            if 'chunked' in self.headers.get('Transfer-Encoding','').lower():
                chunks=[]
                while True:
                    size=int(self.rfile.readline().split(b';',1)[0].strip(),16)
                    if size==0:
                        while self.rfile.readline().strip(): pass
                        break
                    chunks.append(self.rfile.read(size))
                    assert self.rfile.read(2)==b'\r\n'
                body=b''.join(chunks)
            else:
                body=self.rfile.read(int(self.headers.get('Content-Length',0)))
            if measured:
                with mutex:
                    sequence+=1
                    row={'id':f'observer-{sequence:05d}','startedUtc':stamp(),'inputTokens':None,'outputTokens':None,'prefillSeconds':None,'decodeSeconds':None,'clientTtftMs':None,'ttftBoundary':'client-send-to-first-valid-output-token','firstTokenReceived':False,'outcome':'incomplete','workerResult':None}
                    active=row
            target=urlsplit(CLONE) if is_outer else urlsplit(WORKER)
            conn=http.client.HTTPConnection(target.hostname,target.port,timeout=330)
            headers={k:v for k,v in self.headers.items() if k.lower() in ['content-type','authorization','x-api-key','x-deepinfer-token','accept']}
            headers['Content-Length']=str(len(body))
            start=time.monotonic()
            conn.request(self.command,self.path,body=body if body else None,headers=headers)
            response=conn.getresponse()
            self.send_response(response.status)
            self.send_header('Content-Type',response.getheader('Content-Type','application/json'))
            self.send_header('Connection','close');self.end_headers();self.close_connection=True
            stream='text/event-stream' in response.getheader('Content-Type','')
            if not stream:
                self.wfile.write(response.read());self.wfile.flush()
            else:
                while True:
                    line=response.readline()
                    if not line: break
                    received=time.monotonic()
                    if line.startswith(b'data:') and line[5:].strip()!=b'[DONE]':
                        try: event=json.loads(line[5:])
                        except (ValueError,UnicodeError): event={}
                        if is_outer and row is not None:
                            choices=event.get('choices') or []
                            delta=(choices[0].get('delta') or {}) if choices else {}
                            valid=bool(delta.get('content')) or any(bool((v.get('function') or {}).get('name')) or bool((v.get('function') or {}).get('arguments')) for v in delta.get('tool_calls') or [])
                            if valid and row['clientTtftMs'] is None:
                                row['clientTtftMs']=(received-start)*1000;row['firstTokenReceived']=True
                        elif not is_outer and event.get('result'):
                            result=event['result'];timing=result.get('timings') or {}
                            with mutex:
                                if active is not None:
                                    active.update({'inputTokens':result.get('promptTokens'),'outputTokens':len(result.get('tokenIds') or result.get('tokens') or []),'prefillSeconds':timing.get('prefillMs',0)/1000 if 'prefillMs' in timing else None,'decodeSeconds':timing.get('decodeMs',0)/1000 if timing.get('decodeTokens',0)>0 else None,'workerResult':{'requestId':result.get('requestId'),'workerId':result.get('workerId'),'timings':timing,'executionPath':result.get('executionPath'),'finishReason':result.get('finishReason')}})
                    self.wfile.write(line);self.wfile.flush()
            if row is not None:
                row['httpStatus']=response.status
                row['outcome']='completed' if response.status==200 and row['workerResult'] is not None and row['firstTokenReceived'] else 'failed'
        except Exception as e:
            if row is not None: row['errorType']=type(e).__name__;row['outcome']='failed'
            try:
                self.send_response(502);self.send_header('Content-Length','0');self.end_headers()
            except Exception: pass
        finally:
            if conn: conn.close()
            if measured:
                row['completedUtc']=stamp();row['observedSeconds']=time.monotonic()-start
                with mutex:
                    with (OUT/'requests.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                    active=None
                gate.release()

log=(OUT/'coordinator.log').open('w')
coordinator=subprocess.Popen([str(EXE),'coordinator','--config',str(PRIVATE/'coordinator.json'),'--listen',CLONE],stdout=log,stderr=subprocess.STDOUT)
servers=[ThreadingHTTPServer(('127.0.0.1',p),Handler) for p in [args.outer_port,args.worker_port]]
for server in servers: threading.Thread(target=server.serve_forever,daemon=True).start()

def heartbeat():
    while not stop.is_set():
        try:
            snapshot=call(CONFIG['listen']+'/v1/workers')
            registration=snapshot['workers'][0]['registration']
            registration['endpoint']=f'http://127.0.0.1:{args.worker_port}'
            call(CLONE+'/v1/workers/register',registration)
        except Exception: pass
        stop.wait(2)
threading.Thread(target=heartbeat,daemon=True).start()
def shutdown(*args): stop.set()
signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
(OUT/'manifest.json').write_text(json.dumps({'observerCoordinatorBuild':subprocess.check_output([str(EXE),'version'],text=True).strip(),'ports':[args.outer_port,args.clone_port,args.worker_port],'ttftBoundary':'Observer HTTP client immediately before forwarding to the actual main coordinator until first valid text/tool-call output received; role/usage/keepalive ignored. Pi-to-observer dispatch is outside this interval.','workerTimingBoundary':'Raw ServiceEvent.result.timings, copied from real worker stream; no dashboard polling or wall-time inference.','overhead':'Two loopback streaming observation hops, same .NET coordinator/worker/backend code; measured as separate configuration.','promptsAndOutputTextSaved':False,'createdUtc':stamp()},indent=2))
print(f'Observer ready: loopback {args.outer_port}; no prompt/output text logged.',flush=True)
stop.wait()
for server in servers: server.shutdown();server.server_close()
coordinator.terminate();coordinator.wait(timeout=15);log.close()
