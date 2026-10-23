#!/usr/bin/env python3
"""Authenticated direct TCP transport probe with actual CUDA/Metal memory.

Synthetic bit patterns occupy the same byte counts as the model's BF16 KV and
FP32 recurrent state. This does not export/import a running model session.
Control uses SSH only to start the peer; payload travels directly over LAN TCP.
"""
import argparse
import base64
import ctypes
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 ** 2
KV_BYTES = 5 * 2 * 131072 * 2 * 256 * 2
RECURRENT_BYTES = 15 * (32 * 128 * 128 + 3 * 8192) * 4
MAX_BYTES = KV_BYTES + RECURRENT_BYTES
MAX_CHUNK = 16 * MIB


def exact(sock, length):
    data = bytearray(length)
    receive(sock, memoryview(data))
    return data


def receive(sock, view):
    offset = 0
    while offset < len(view):
        n = sock.recv_into(view[offset:])
        if not n:
            raise EOFError('Peer disconnected during payload')
        offset += n


def send_json(sock, obj):
    raw = json.dumps(obj, separators=(',', ':')).encode()
    if len(raw) > 65536:
        raise ValueError('Oversized control message')
    sock.sendall(struct.pack('!I', len(raw)) + raw)


def recv_json(sock):
    length, = struct.unpack('!I', exact(sock, 4))
    if not 0 < length <= 65536:
        raise ValueError('Invalid control frame length')
    return json.loads(exact(sock, length))


class Gpu:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        l = self.lib
        l.st_open.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        l.st_prepare.argtypes = [ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
        l.st_pointer.argtypes = [ctypes.c_size_t]
        l.st_pointer.restype = ctypes.c_void_p
        for name in ['st_read', 'st_write']:
            getattr(l, name).argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        l.st_validate.argtypes = [ctypes.c_size_t, ctypes.c_uint32]
        l.st_error.restype = ctypes.c_char_p
        self.check(l.st_open(MAX_BYTES, MAX_CHUNK))

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.st_error().decode() or 'GPU operation failed')

    def view(self, offset, length):
        return memoryview((ctypes.c_ubyte * length).from_address(self.lib.st_pointer(offset))).cast('B')

    def selftest(self):
        # Check pattern against a CPU reference, tails, and an intentional fault.
        size, seed = 1028, 0x12345678
        self.check(self.lib.st_prepare(size, seed, 1))
        for offset in range(0, size, 256):
            count = min(256, size-offset)
            self.check(self.lib.st_read(offset, count))
            expected = b''.join(struct.pack('<I', (((i*1664525+1013904223)&0xffffffff)^seed))
                                for i in range(offset//4, (offset+count)//4))
            assert bytes(self.view(offset, count)) == expected
        self.check(self.lib.st_validate(size, seed))
        self.check(self.lib.st_read(0, 4))
        self.view(0, 4)[0] ^= 1
        self.check(self.lib.st_write(0, 4))
        assert self.lib.st_validate(size, seed) != 0, 'Corruption went undetected'


def metadata(library):
    return {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'librarySHA256': hashlib.sha256(Path(library).read_bytes()).hexdigest(),
            'python': sys.version, 'platform': sys.platform}


def prepare(gpu, case, is_source):
    n, seed = case['bytes'], case['seed']
    if case['mode'] == 'gpu':
        gpu.check(gpu.lib.st_prepare(n, seed, int(is_source)))
        return None
    # Deterministic nonzero host data; no compression on this TCP channel.
    pattern = bytes((i ^ (seed & 255)) for i in range(256))
    return bytearray((pattern * ((n+255)//256))[:n]) if is_source else bytearray(n)


def transfer(sock, gpu, case, source, host):
    n, chunk = case['bytes'], case['chunk']
    staging = 0.0
    started = time.perf_counter()
    for offset in range(0, n, chunk):
        length = min(chunk, n-offset)
        if host is None:
            view = gpu.view(offset, length)
        else:
            view = memoryview(host)[offset:offset+length]
        if source:
            if host is None:
                t = time.perf_counter(); gpu.check(gpu.lib.st_read(offset, length)); staging += time.perf_counter()-t
            sock.sendall(view)
        else:
            receive(sock, view)
            if host is None:
                t = time.perf_counter(); gpu.check(gpu.lib.st_write(offset, length)); staging += time.perf_counter()-t
    payload_seconds = time.perf_counter()-started
    validation = time.perf_counter()
    if not source:
        if host is None:
            gpu.check(gpu.lib.st_validate(n, case['seed']))
        else:
            assert zlib.crc32(host) == case['crc32'], 'Host payload checksum mismatch'
    validation_seconds = time.perf_counter()-validation
    return {'payloadSeconds': payload_seconds, 'stagingSeconds': staging,
            'validationSeconds': validation_seconds, 'validated': not source}


def validate_case(case):
    assert case['mode'] in ('gpu', 'host')
    assert case['direction'] in ('cuda-to-metal', 'metal-to-cuda')
    assert isinstance(case['bytes'], int) and 0 < case['bytes'] <= MAX_BYTES and case['bytes'] % 4 == 0
    assert isinstance(case['chunk'], int) and 0 < case['chunk'] <= MAX_CHUNK and case['chunk'] % 4 == 0


def client(args):
    config = json.loads(sys.stdin.readline())  # secret delivered through SSH stdin
    library = ROOT / 'artifacts/state-transfer/transfer_cuda.dll'
    run_id = config['runId']
    assert run_id and all(c.isalnum() or c in '-_' for c in run_id)
    remote_out = ROOT / 'artifacts/state-transfer' / run_id
    remote_out.mkdir(exist_ok=False)
    with socket.create_connection((config['host'], config['port']), timeout=30) as sock:
        sock.settimeout(60)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        challenge = recv_json(sock)['challenge']
        signature = hmac.new(bytes.fromhex(config['token']), challenge.encode(), hashlib.sha256).hexdigest()
        send_json(sock, {'proof': signature, **metadata(library)})
        assert recv_json(sock)['authenticated']
        gpu = Gpu(library)
        try:
            gpu.selftest()
            send_json(sock, {'selftest': 'passed'})
            while True:
                case = recv_json(sock)
                if case.get('stop'):
                    break
                validate_case(case)
                source = case['direction'] == 'cuda-to-metal'
                host = prepare(gpu, case, source)
                send_json(sock, {'ready': True})
                assert recv_json(sock)['go']
                stats = transfer(sock, gpu, case, source, host)
                if source:
                    assert recv_json(sock)['received']
                send_json(sock, stats)
                record = json.dumps({'case': case, 'windows': stats})
                with (remote_out/'trials.jsonl').open('a') as saved:
                    saved.write(record+'\n')
                print(record, flush=True)
                del host
        finally:
            gpu.lib.st_close()
        (remote_out/'completed.json').write_text(json.dumps(metadata(library)))


def cases():
    for chunk in (256*1024, 4*MIB, 16*MIB):
        yield {'name': 'host-128MiB', 'mode': 'host', 'bytes': 128*MIB, 'chunk': chunk}
    yield {'name': '128K-first20-state', 'mode': 'gpu', 'bytes': MAX_BYTES, 'chunk': 4*MIB}
    # 90% already synchronized at both ends; round suffix up to 13108 tokens.
    yield {'name': '90pct-hit-state-delta', 'mode': 'gpu', 'bytes': 13108*10240+RECURRENT_BYTES, 'chunk': 4*MIB}


def server(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    library = ROOT / 'artifacts/state-transfer/transfer_metal.dylib'
    manifest = {'schemaVersion': 1, 'kind': 'synthetic-state-transport', 'transport': 'direct TCP, authenticated; no TLS or compression; SSH control only',
                'modelExecution': False, 'fullKVBytes': KV_BYTES, 'recurrentBytes': RECURRENT_BYTES,
                'local': metadata(library), 'warmups': 1, 'trials': 3,
                'timing': 'Mac monotonic clock; GO through receiver GPU validation and final acknowledgment; allocation/pattern fill excluded',
                'gpuPath': 'CUDA device <-> pinned staging <-> socket <-> Metal shared MTLBuffer; no overlap/double buffering',
                'status': 'running'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # Bind only the interface that routes to the explicitly configured SSH peer.
    peer = subprocess.check_output(['ssh', '-G', args.ssh], text=True)
    remote_ip = next(line.split()[1] for line in peer.splitlines() if line.startswith('hostname '))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect((remote_ip, 22)); local_ip = route.getsockname()[0]
    secret = secrets.token_bytes(32)
    proc = None
    gpu = None
    try:
        with socket.socket() as listener, (out/'windows.log').open('w') as log, (out/'trials.jsonl').open('w') as raw:
            listener.bind((local_ip, 0)); listener.listen(1); listener.settimeout(45)
            script = r"Set-Location C:\Users\xiaoyun\repos\Infer.NET; & \"$env:LOCALAPPDATA\Infer.NET\Python\python.exe\" -u scripts\benchmark-state-transfer.py client".replace('\\"','"')
            encoded = base64.b64encode(script.encode('utf-16le')).decode()
            proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', args.ssh,
                                     'powershell.exe -NoProfile -NonInteractive -EncodedCommand '+encoded],
                                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
            proc.stdin.write(json.dumps({'host': local_ip, 'port': listener.getsockname()[1], 'token': secret.hex(), 'runId': out.name})+'\n')
            proc.stdin.close()
            sock, address = listener.accept()
            with sock:
                assert address[0] == socket.gethostbyname(remote_ip), 'Unexpected peer address'
                sock.settimeout(60); sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                nonce = secrets.token_hex(32); send_json(sock, {'challenge': nonce})
                hello = recv_json(sock)
                assert hmac.compare_digest(hello.pop('proof'), hmac.new(secret, nonce.encode(), hashlib.sha256).hexdigest()), 'Authentication failed'
                assert hello['commit'] == manifest['local']['commit'], 'Source commit mismatch'
                manifest['remote'] = hello
                send_json(sock, {'authenticated': True})
                gpu = Gpu(library); gpu.selftest()
                assert recv_json(sock)['selftest'] == 'passed'
                manifest['gpuSelftests'] = 'CPU pattern, tail and corruption rejection passed on both GPUs'
                for config in cases():
                    for direction in ('cuda-to-metal', 'metal-to-cuda'):
                        for trial in range(4):
                            case = {**config, 'direction': direction, 'trial': trial, 'warmup': trial == 0, 'seed': 1000+trial}
                            source = direction == 'metal-to-cuda'
                            host = prepare(gpu, case, source)
                            if case['mode'] == 'host':
                                reference = host if source else prepare(gpu, case, True)
                                case['crc32'] = zlib.crc32(reference)
                                del reference
                            validate_case(case)
                            send_json(sock, case); assert recv_json(sock)['ready']
                            started = time.perf_counter(); send_json(sock, {'go': True})
                            local_stats = transfer(sock, gpu, case, source, host)
                            if not source:
                                send_json(sock, {'received': True})
                            remote_stats = recv_json(sock)
                            elapsed = time.perf_counter()-started
                            record = {**case, 'seconds': elapsed, 'MBps': case['bytes']/elapsed/1e6,
                                      'mac': local_stats, 'windows': remote_stats}
                            raw.write(json.dumps(record)+'\n'); raw.flush()
                            print(json.dumps(record), flush=True)
                            del host
                send_json(sock, {'stop': True})
            proc.wait(timeout=30)
            assert proc.returncode == 0, 'Remote process failed'
        manifest['status'] = 'passed'
    except BaseException as exc:
        manifest['status'] = 'failed'; manifest['error'] = str(exc)
        raise
    finally:
        if gpu: gpu.lib.st_close()
        if proc and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=['server', 'client'])
    parser.add_argument('--ssh', default='infer-windows')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.role == 'server' and not args.output: parser.error('--output required')
    (server if args.role == 'server' else client)(args)
