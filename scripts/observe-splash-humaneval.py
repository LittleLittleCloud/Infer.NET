#!/usr/bin/env python3
"""Observe serial Pi requests to Splash without persisting prompts or answers.

Native command counters and block-emission lifecycle metrics are separate.
Neither is presented as Infer.NET's first-to-last-token N-1 decode timing.
"""
import argparse
import datetime
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request
import uuid


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def valid_output(event):
    for choice in event.get('choices') or []:
        delta = choice.get('delta') or {}
        if delta.get('content') or delta.get('reasoning_content'):
            return True
        if any((tool.get('function') or {}).get('name') or
               (tool.get('function') or {}).get('arguments')
               for tool in delta.get('tool_calls') or []):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', type=int, default=5288)
    parser.add_argument('--upstream-port', type=int, default=5289)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gate, write_lock, stop = threading.Lock(), threading.Lock(), threading.Event()

    def save(name, row):
        with write_lock, (args.output / name).open('a') as stream:
            stream.write(json.dumps(row) + '\n')

    def status():
        with urllib.request.urlopen(f'http://127.0.0.1:{args.upstream_port}/status', timeout=10) as response:
            return json.load(response)

    def resource():
        row = {'utc': stamp()}
        for name, command in [('vm_stat', ['vm_stat']), ('power', ['pmset', '-g', 'batt'])]:
            row[name] = subprocess.check_output(command, text=True)
        try:
            row['status'] = status()
        except Exception as error:
            row['statusError'] = type(error).__name__
        save('resources.jsonl', row)

    def sample():
        while not stop.is_set():
            resource()
            stop.wait(10)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.forward(False)

        def do_POST(self):
            self.forward(self.path == '/v1/chat/completions')

        def forward(self, measured):
            row, connection = None, None
            if measured:
                gate.acquire()
            try:
                if 'chunked' in self.headers.get('Transfer-Encoding', '').lower():
                    chunks = []
                    while True:
                        size = int(self.rfile.readline().split(b';')[0], 16)
                        if not size:
                            while self.rfile.readline().strip():
                                pass
                            break
                        chunks.append(self.rfile.read(size))
                        assert self.rfile.read(2) == b'\r\n'
                    body = b''.join(chunks)
                else:
                    body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                if measured:
                    request = json.loads(body)
                    # Documented upstream namespace isolation: no prompt edits.
                    request['cache_salt'] = str(uuid.uuid4())
                    request['temperature'] = 0
                    request['reasoning_effort'] = 'none'
                    request['stream_options'] = {'include_usage': True}
                    body = json.dumps(request).encode()
                    row = {'id': str(uuid.uuid4()), 'startedUtc': stamp(),
                           'before': status(), 'usage': None, 'metrics': None,
                           'timings': None, 'clientTtftMs': None,
                           'firstTokenReceived': False, 'outcome': 'incomplete'}
                connection = http.client.HTTPConnection('127.0.0.1', args.upstream_port, timeout=330)
                start = time.monotonic()
                connection.request(self.command, self.path, body=body or None,
                                   headers={'Content-Type': 'application/json',
                                            'Content-Length': str(len(body))})
                response = connection.getresponse()
                self.send_response(response.status)
                self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.close_connection = True
                if row is not None:
                    row['httpStatus'] = response.status
                if 'text/event-stream' in response.getheader('Content-Type', ''):
                    while line := response.readline():
                        received = time.monotonic()
                        if measured and line.startswith(b'data:'):
                            payload = line[5:].strip()
                            if payload != b'[DONE]':
                                event = json.loads(payload)
                                if valid_output(event) and row['clientTtftMs'] is None:
                                    row['clientTtftMs'] = (received-start)*1000
                                    row['firstTokenReceived'] = True
                                for key in ('usage', 'metrics', 'timings'):
                                    if event.get(key) is not None:
                                        row[key] = event[key]
                        self.wfile.write(line)
                        self.wfile.flush()
                else:
                    self.wfile.write(response.read())
                    self.wfile.flush()
                if row is not None:
                    row['outcome'] = 'completed' if response.status == 200 and row['usage'] else 'failed'
            except Exception as error:
                if row is not None:
                    row['outcome'] = 'failed'
                    row['errorType'] = type(error).__name__
                try:
                    self.send_error(502)
                except OSError:
                    pass
            finally:
                if connection:
                    connection.close()
                if measured:
                    if row is not None:
                        row['completedUtc'] = stamp()
                        try:
                            row['after'] = status()
                        except Exception as error:
                            row['afterError'] = type(error).__name__
                        save('requests.jsonl', row)
                    gate.release()

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.daemon_threads = True
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stop.set())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    print('Splash observer ready; cold namespaces, greedy, reasoning off.', flush=True)
    stop.wait()
    server.shutdown()
    server.server_close()
    sampler.join(timeout=15)
    resource()


if __name__ == '__main__':
    main()
