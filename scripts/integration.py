#!/usr/bin/env python3
"""Real processes and real HTTP. No external Python dependencies. Run after building."""
import argparse, concurrent.futures, json, os, pathlib, signal, socket, subprocess, tempfile, time
import urllib.request, urllib.error
import http.client
import http.server, threading
from contextlib import closing
from urllib.parse import urlsplit

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--executable', type=pathlib.Path, default=ROOT/'src/DeepInfer.Host/bin/Release/net10.0/deepinfer')
args = parser.parse_args()
exe = str(args.executable.resolve())
processes, logs = [], []
checks = 0

def check(condition, message):
    global checks
    assert condition, message
    checks += 1
    print('PASS:', message, flush=True)

def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def start(*arguments):
    log = tempfile.TemporaryFile(mode='w+')
    env = {**os.environ, 'DEEPINFER_TOKEN': 'integration-test-secret'}
    p = subprocess.Popen([exe, *arguments], stdout=log, stderr=log, env=env)
    processes.append(p); logs.append(log)
    return p

def rpc(origin, path, data=None, token='integration-test-secret'):
    request = urllib.request.Request(origin + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type':'application/json', 'X-DeepInfer-Token':token})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            content = response.read()
            return response.status, json.loads(content) if content else None
    except urllib.error.HTTPError as error:
        content = error.read()
        try: content = json.loads(content)
        except ValueError: pass
        return error.code, content

def wait_for(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value: return value
        except (urllib.error.URLError, TimeoutError, ConnectionError): pass
        time.sleep(.08)
    raise AssertionError('Timed out waiting for cluster state')

def registration(id, endpoint, version=1):
    return {'protocolVersion':version,'workerId':id,'endpoint':endpoint,'capacity':1,
        'devices':[{'id':'sim:0','backend':'simulation','name':'test','architecture':'test','memoryBytes':None,
                    'nativeAvailable':True,'supportsInference':True,'status':'synthetic'}],
        'models':[{'id':'synthetic-v1','format':'synthetic','isSimulation':True}]}

try:
    option = subprocess.run([exe, 'version', '--metal-residency', 'off', '--residency-refresh-ms', '0', '--short-prefill-tokens', '128', '--cuda-readback', 'batch', '--cuda-prefill', 'batch-fp32'], capture_output=True, text=True, timeout=10)
    check(option.returncode == 0, 'residency A/B CLI options are accepted')
    for arguments, expected in [(['--metal-residency', 'invalid'], 'metal-residency must be on or off'), (['--residency-refresh-ms', '-1'], 'residency-refresh-ms must be 0..60000'), (['--short-prefill-tokens', '129'], 'short-prefill-tokens must be 0..128'), (['--cuda-readback', 'invalid'], 'cuda-readback must be batch or token'), (['--cuda-prefill', 'invalid'], 'cuda-prefill must be serial or batch-fp32'), (['--suffix-chunk-tokens', '64'], 'suffix-chunk-tokens must be 32, 128 or 512'), (['--cuda-send-overlap', 'invalid'], 'cuda-send-overlap must be on or off')]:
        option = subprocess.run([exe, 'split-serve', *arguments], capture_output=True, text=True, timeout=10)
        check(option.returncode != 0 and expected in option.stdout + option.stderr, 'residency invalid policy rejected before model load: ' + arguments[0])
    origin = f'http://127.0.0.1:{port()}'
    start('coordinator', '--listen', origin)
    wait_for(lambda: rpc(origin, '/health', token='')[0] == 200)
    check(rpc(origin, '/health', token='')[1]['protocolVersion'] == 1, 'health and protocol version')
    check(rpc(origin, '/v1/workers', token='wrong')[0] == 401, 'authentication rejects bad token')
    request = {'prompt':'你好 distributed world','maxTokens':12,'model':'synthetic-v1','backend':'simulation'}
    check(rpc(origin, '/v1/infer', request)[0] == 503, 'empty cluster returns 503')
    check(rpc(origin, '/v1/workers/register', registration('bad', 'http://127.0.0.1:1', 99))[0] == 400, 'version mismatch rejected')
    check(rpc(origin, '/v1/workers/register', registration('bad', 'http://example.com'))[0] == 400, 'remote registration disabled by default')
    bad = registration('bad', 'http://127.0.0.1:1'); bad['devices'] = [None]
    check(rpc(origin, '/v1/workers/register', bad)[0] == 400, 'null device advertisement rejected')
    workers = []
    for name in ('one', 'two'):
        endpoint = f'http://127.0.0.1:{port()}'
        workers.append((start('worker','--id',name,'--listen',endpoint,'--coordinator',origin), endpoint))
    wait_for(lambda: len(rpc(origin, '/v1/workers')[1]['workers']) == 2)
    responses = [rpc(origin, '/v1/infer', request) for _ in range(6)]
    check(all(code == 200 for code, _ in responses), 'HTTP RPC inference succeeds')
    results = [r for _, r in responses]
    check([r['workerId'] for r in results] == ['one','two']*3, 'round robin crosses independent processes')
    check(all(r['isSimulation'] and r['text'].startswith('[SIMULATION]') and len(r['tokens']) == 12 for r in results), 'synthetic results explicitly labelled and bounded')
    check(all(r['tokens'] == results[0]['tokens'] for r in results), 'deterministic output across worker processes')
    for change in ({'maxTokens':0}, {'prompt':''}, {'prompt':'a'*32769}):
        check(rpc(origin, '/v1/infer', {**request, **change})[0] == 400, f'invalid request rejected: {next(iter(change))}')
    check(rpc(origin, '/v1/infer', {**request,'backend':'metal'})[0] == 503, 'GPU request never falls back to simulation')
    check(rpc(workers[0][1], '/v1/infer', {**request,'backend':'cuda'})[0] == 501, 'direct worker rejects unavailable GPU')
    check(rpc(origin, '/v1/infer', {**request,'prompt':'x'*300000})[0] == 400, 'legacy model prompt bound remains enforced with larger HTTP envelope')
    u = urlsplit(origin)
    with closing(http.client.HTTPConnection(u.hostname, u.port, timeout=5)) as connection:
        connection.request('POST', '/v1/infer', headers={'Content-Type':'application/json',
            'Content-Length':str(64*1024*1024+1), 'X-DeepInfer-Token':'integration-test-secret'})
        check(connection.getresponse().status == 413, '64 MiB HTTP request body limit enforced before upload')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(lambda _: rpc(origin, '/v1/infer', request)[0], range(32)))
    check(all(code in (200,429) for code in codes) and 200 in codes, 'concurrent requests complete or receive explicit backpressure')
    benchmark = subprocess.run([exe,'bench','--endpoint',origin,'--requests','10','--warmup','2','--backend','simulation','--model','synthetic-v1','--max-tokens','16'],
        env={**os.environ,'DEEPINFER_TOKEN':'integration-test-secret'},text=True,capture_output=True,timeout=10)
    check(benchmark.returncode == 0 and json.loads(benchmark.stdout)['outputUnits'] == 160, 'benchmark CLI exercises actual RPC')
    # Before lease expiry a dead worker returns 502; after expiry it is no longer eligible.
    workers[0][0].kill(); workers[0][0].wait(timeout=5)
    codes = [rpc(origin, '/v1/infer', request)[0] for _ in range(2)]
    check(502 in codes and 200 in codes, 'dead worker produces 502 while surviving worker still serves')
    wait_for(lambda: len([w for w in rpc(origin,'/v1/workers')[1]['workers'] if w['healthy']]) == 1, timeout=11)
    check(all(rpc(origin,'/v1/infer',request)[1]['workerId'] == 'two' for _ in range(4)), 'expired lease removes dead worker from scheduling')
    # A real Metal worker advertises hardware but never advertises a loaded LLM.
    gpu_endpoint = f'http://127.0.0.1:{port()}'
    start('worker','--id','gpu-stub','--backend','metal','--listen',gpu_endpoint,'--coordinator',origin)
    gpu = wait_for(lambda: next((w for w in rpc(origin,'/v1/workers')[1]['workers'] if w['registration']['workerId']=='gpu-stub'), None))
    check(gpu['registration']['models'] == [], 'Metal worker without weights advertises no loaded model')
    check(rpc(gpu_endpoint,'/v1/infer',{**request,'backend':'metal'})[0] == 501, 'Metal worker explicitly rejects inference')
    workers[1][0].kill(); workers[1][0].wait(timeout=5)
    wait_for(lambda: not any(w['healthy'] and w['registration']['workerId']=='two' for w in rpc(origin,'/v1/workers')[1]['workers']),timeout=11)
    check(rpc(origin,'/v1/infer',request)[0] == 503, 'no live implemented model returns 503')
    # A locally ready split worker can still lack the CUDA peer for a long prompt.
    # Preserve its explicit unavailable/timeout status, distinct from a dead socket.
    class UnavailableWorker(http.server.BaseHTTPRequestHandler):
        status = 503
        def do_POST(self):
            # Deliberately reject before consuming the body (HttpClient may stream chunked).
            body = json.dumps({'code':'not_ready','message':'CUDA peer unavailable'}).encode()
            self.send_response(self.status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body))); self.send_header('Connection','close')
            self.end_headers(); self.wfile.write(body); self.close_connection = True
        def log_message(self, *args): pass
    with http.server.ThreadingHTTPServer(('127.0.0.1',0), UnavailableWorker) as unavailable:
        thread = threading.Thread(target=unavailable.serve_forever, daemon=True); thread.start()
        try:
            endpoint = f'http://127.0.0.1:{unavailable.server_port}'
            check(rpc(origin,'/v1/workers/register',registration('unavailable',endpoint))[0] == 204, 'register locally ready worker fixture')
            for status in (503,504):
                UnavailableWorker.status = status
                check(rpc(origin,'/v1/infer',request)[0] == status, f'coordinator preserves worker {status}')
        finally: unavailable.shutdown(); thread.join(timeout=5)
    coordinator = processes[0]
    coordinator.send_signal(signal.SIGTERM); coordinator.wait(timeout=5)
    rejoin_endpoint = f'http://127.0.0.1:{port()}'
    start('worker','--id','rejoining','--listen',rejoin_endpoint,'--coordinator',origin)
    time.sleep(.2)
    start('coordinator','--listen',origin)
    wait_for(lambda: any(w['registration']['workerId']=='rejoining' for w in rpc(origin,'/v1/workers')[1]['workers']))
    check(rpc(origin,'/v1/infer',request)[1]['workerId']=='rejoining', 'worker retries registration and serves after coordinator restart')
    print(f'{checks} integration checks passed: {exe}', flush=True)
finally:
    for p in reversed(processes):
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired: p.kill(); p.wait(timeout=5)
    if __import__('sys').exc_info()[0]:
        for log in logs:
            log.seek(0); print(log.read())
    for log in logs: log.close()
