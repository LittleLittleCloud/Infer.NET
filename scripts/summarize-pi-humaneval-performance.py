#!/usr/bin/env python3
"""Reconcile terminal worker clocks with every Pi request; never infer GPU time."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re


def instant(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def summarize(run, log, mac_commit, windows_commit=None, change=''):
    if not re.fullmatch('[0-9a-f]{40}', mac_commit):
        raise ValueError('Require verified full Mac source SHA')
    if windows_commit is not None and not re.fullmatch('[0-9a-f]{40}', windows_commit):
        raise ValueError('Require verified full Windows source SHA')
    tasks = [json.loads(line) for line in (run / 'tasks.jsonl').read_text().splitlines() if line.strip()]
    metrics = []
    for line in log.read_text().splitlines():
        if 'INFERENCE_METRIC ' in line:
            metrics.append(json.loads(line.split('INFERENCE_METRIC ', 1)[1]))
    selected, problems, per_task = [], [], []
    for task in tasks:
        start, end = instant(task['start_utc']), instant(task['end_utc'])
        clocks = [m for m in metrics if start <= instant(m['completedAt']) <= end]
        expected = task['pi']['requests']
        if len(clocks) != len(expected):
            problems.append(f"{task['task_id']}: {len(clocks)}/{len(expected)} terminal clocks")
        for worker, pi in zip(clocks, expected):
            if mac_commit not in worker['build']:
                problems.append(f"{task['task_id']}: worker build mismatch")
            if (worker['inputTokens'], worker['outputTokens']) != (pi['usage'].get('input'), pi['usage'].get('output')):
                problems.append(f"{task['task_id']}: worker/Pi token mismatch")
            if worker['decodeIntervals'] != max(0, worker['outputTokens'] - 1):
                problems.append(f"{task['task_id']}: invalid N-1 interval count")
            for key in ('prefillMs', 'decodeMs'):
                if not math.isfinite(worker[key]) or worker[key] < 0:
                    problems.append(f"{task['task_id']}: invalid {key}")
            if worker['prefillMs'] <= 0 or (worker['decodeIntervals'] and worker['decodeMs'] <= 0):
                problems.append(f"{task['task_id']}: missing worker duration")
        selected.extend(clocks)
        per_task.append({'task_id': task['task_id'], 'expected_requests': len(expected),
                         'observed_requests': len(clocks), 'completed': task.get('completed'),
                         'timeout': task.get('timeout'), 'worker_metrics': clocks})
    if len({m['requestId'] for m in selected}) != len(selected):
        problems.append('Duplicate or multiply assigned worker request IDs')
    if len({(m['executionPath'], m['kv']) for m in selected}) > 1:
        problems.append('Mixed execution paths or KV profiles')
    expected_count = sum(len(t['pi']['requests']) for t in tasks)
    prefill_ms = sum(m['prefillMs'] for m in selected)
    decode_ms = sum(m['decodeMs'] for m in selected)
    complete = not problems and bool(tasks) and len(selected) == expected_count and bool(selected)
    return {
        'schema': 'infernet-worker-performance-v1', 'mac_commit': mac_commit,
        'windows_commit': windows_commit, 'change': change, 'metrics_complete': complete,
        'expected_requests': expected_count, 'observed_requests': len(selected), 'problems': problems,
        'input_tokens': sum(m['inputTokens'] for m in selected),
        'output_tokens': sum(m['outputTokens'] for m in selected),
        'decode_intervals': sum(m['decodeIntervals'] for m in selected),
        'prefill_ms': prefill_ms, 'decode_ms': decode_ms,
        'prefill_tokens_per_second': sum(m['inputTokens'] for m in selected) / (prefill_ms / 1000) if complete and prefill_ms > 0 else None,
        'decode_tokens_per_second': sum(m['decodeIntervals'] for m in selected) / (decode_ms / 1000) if complete and decode_ms > 0 else None,
        'client_ttft_ms': None, 'client_ttft_note': 'Requires an independent first-valid-token observer.',
        'tasks': per_task,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--serve-log', type=Path, required=True)
    p.add_argument('--mac-commit', required=True)
    p.add_argument('--windows-commit')
    p.add_argument('--change', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.run, a.serve_log, a.mac_commit, a.windows_commit, a.change)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('metrics_complete', 'expected_requests', 'observed_requests', 'prefill_tokens_per_second', 'decode_tokens_per_second', 'problems')}))


if __name__ == '__main__':
    main()
