#!/usr/bin/env python3
"""macOS/Linux PTY smoke test: RazorConsole renders live worker dashboard RPC."""
import argparse, errno, fcntl, json, os, pathlib, pty, re, select, signal, socket, struct, subprocess, termios, time, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--host',type=pathlib.Path,default=ROOT/'src/DeepInfer.Host/bin/Release/net10.0/deepinfer')
p.add_argument('--tui',type=pathlib.Path,default=ROOT/'src/DeepInfer.Tui/bin/Release/net10.0/deepinfer-tui')
p.add_argument('--qwen36',action='store_true')
a=p.parse_args()
# Demo reserves consecutive ports; pick a range by checking all three before launch.
for _ in range(50):
    sockets=[]
    try:
        s=socket.socket();s.bind(('127.0.0.1',0));base=s.getsockname()[1];sockets.append(s)
        for port in (base+1,base+2):
            s=socket.socket();s.bind(('127.0.0.1',port));sockets.append(s)
        break
    except OSError: pass
    finally:
        for s in sockets:s.close()
else: raise RuntimeError('No free port range')
env={**os.environ,'TERM':'xterm-256color'}
env.pop('DEEPINFER_TOKEN',None)
import tempfile
cluster_log=tempfile.TemporaryFile(mode='w+')
cluster_args=[str(a.host.resolve()),'demo','--base-port',str(base),'--stay','--backend','simulation']
if a.qwen36:
    cluster_args=[str(a.host.resolve()),'demo','--base-port',str(base),'--stay','--backend','metal','--model','qwen3.6-35b-a3b','--weights',str(ROOT/'models/qwen3.6-35b-a3b-8bit-mlx'),'--workers','1']
cluster=subprocess.Popen(cluster_args,env=env,stdout=cluster_log,stderr=cluster_log,start_new_session=True)
child=None;master=None;output=b''
try:
    for _ in range(300):
        if cluster.poll() is not None: raise RuntimeError('Cluster exited before ready')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{base}/v1/workers',timeout=.3) as r:
                registered=len(json.load(r)['workers'])==(1 if a.qwen36 else 2)
                cluster_log.seek(0)
                if registered and 'Cluster stays online' in cluster_log.read():break
        except (OSError, ValueError):pass
        time.sleep(.1)
    else:raise RuntimeError('Cluster did not become ready')
    child,master=pty.fork()
    if child==0:
        os.execve(str(a.tui.resolve()),[str(a.tui.resolve()),f'http://127.0.0.1:{base}'],env)
    fcntl.ioctl(master,termios.TIOCSWINSZ,struct.pack('HHHH',40,140,0,0))
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        if select.select([master],[],[],.2)[0]:
            try:data=os.read(master,65536)
            except OSError as e:
                if e.errno==errno.EIO: break
                raise
            if not data:break
            output+=data
            # Answer common cursor-position queries made by terminal UI libraries.
            if b'\x1b[6n' in data:os.write(master,b'\x1b[1;1R')
        text=output.decode(errors='replace')
        expected='Workers · 1 live / 1 known' if a.qwen36 else 'Workers · 2 live / 2 known'
        if expected in text and 'Dashboard' in text and 'Playground' in text and 'Devices' in text:break
    text=output.decode(errors='replace')
    artifact=ROOT/'artifacts/tui-smoke.txt';artifact.parent.mkdir(exist_ok=True)
    artifact.write_text(re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text))
    assert 'Infer.NET' in text and 'Dashboard' in text and 'Playground' in text, f'TUI tabs were not rendered; see {artifact}'
    assert expected in text and 'Devices' in text and 'Models' in text, f'TUI worker RPC result missing; see {artifact}'
    assert ('synthetic-v1 [SIMULATION]' if not a.qwen36 else 'metal') in text, f'TUI worker type missing; see {artifact}'
    print('PASS: RazorConsole PTY -> coordinator -> live worker dashboard')
finally:
    if child:
        try:os.kill(child,signal.SIGINT)
        except ProcessLookupError:pass
        for _ in range(30):
            try:
                ended,_=os.waitpid(child,os.WNOHANG)
                if ended:break
            except ChildProcessError:break
            time.sleep(.1)
        else:
            os.kill(child,signal.SIGKILL);os.waitpid(child,0)
    if master is not None:os.close(master)
    # SIGINT triggers the demo's finally block, which owns and reaps its child processes.
    if cluster.poll() is None:
        cluster.send_signal(signal.SIGINT)
        try:cluster.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(cluster.pid,signal.SIGKILL);cluster.wait()
    cluster_log.close()
