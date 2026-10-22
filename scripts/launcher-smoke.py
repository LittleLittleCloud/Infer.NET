#!/usr/bin/env python3
"""Exercise the actual one-click Qwen launcher in a PTY, submit a request, verify shutdown."""
import argparse,fcntl,os,pathlib,pty,re,select,signal,socket,struct,subprocess,termios,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--size',choices=['3b','7b','moe'],default='3b')
a=parser.parse_args()
model='qwen3.6-35b-a3b' if a.size=='moe' else f'qwen2.5-{a.size}-instruct'
command=ROOT/('DeepInferMoE.command' if a.size=='moe' else 'DeepInfer7B.command' if a.size=='7b' else 'DeepInfer.command')
for _ in range(50):
 sockets=[]
 try:
  s=socket.socket();s.bind(('127.0.0.1',0));base=s.getsockname()[1];sockets.append(s)
  s=socket.socket();s.bind(('127.0.0.1',base+1));sockets.append(s);break
 except OSError:pass
 finally:
  for s in sockets:s.close()
else:raise RuntimeError('No free port range')
child,master=pty.fork()
if child==0:
 env={**os.environ,'TERM':'xterm-256color','DEEPINFER_BASE_PORT':str(base)};env.pop('DEEPINFER_TOKEN',None)
 os.execve(str(command),[str(command)],env)
fcntl.ioctl(master,termios.TIOCSWINSZ,struct.pack('HHHH',40,160,0,0))
output=b'';clicked=False;exited=False;owned=set()
try:
 deadline=time.monotonic()+90
 while time.monotonic()<deadline:
  if select.select([master],[],[],.2)[0]:
   try:data=os.read(master,65536)
   except OSError:break
   output+=data
   if b'\x1b[6n' in data:os.write(master,b'\x1b[1;1R')
  text=output.decode(errors='replace')
  if '1 live workers' in text and model in text and not clicked:
   os.write(master,b'\t\r');clicked=True
  if ('Qwen' in text or '阿里' in text or '通义' in text) and 'Backend: metal' in text:break
 assert clicked and ('Qwen' in text or '阿里' in text or '通义' in text) and 'Backend: metal' in text,'Launcher did not generate real Qwen Chinese text'
 processes=subprocess.check_output(['ps','-Ao','pid,ppid,comm'],text=True)
 descendants={child}
 entries=[line.strip().split(None,2) for line in processes.splitlines()[1:]]
 for _ in range(5):
  for pid,parent,command in entries:
   if int(parent) in descendants:descendants.add(int(pid))
 owned={int(pid) for pid,parent,command in entries if int(pid) in descendants and 'deepinfer' in command}
 os.write(master,b'\x03')
 deadline=time.monotonic()+10
 while time.monotonic()<deadline:
  ended,_=os.waitpid(child,os.WNOHANG)
  if ended:exited=True;break
  if select.select([master],[],[],.1)[0]:
   try:os.read(master,65536)
   except OSError:pass
 assert exited,'Launcher did not exit after Ctrl+C'
 current=subprocess.check_output(['ps','-Ao','pid,comm'],text=True)
 leaked=[line for line in current.splitlines()[1:] if int(line.split(None,1)[0]) in owned]
 assert not leaked,'Launcher left owned inference processes running'
 artifact=ROOT/f'artifacts/qwen-{a.size}-launcher-smoke.txt';artifact.parent.mkdir(exist_ok=True)
 artifact.write_text(re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text))
 print('PASS: actual one-click launcher -> Qwen TUI -> independent Metal worker -> Chinese text; Ctrl+C reaped all owned processes')
finally:
 if not exited:
  try:os.killpg(child,signal.SIGKILL)
  except ProcessLookupError:pass
  try:os.waitpid(child,0)
  except ChildProcessError:pass
 os.close(master)
