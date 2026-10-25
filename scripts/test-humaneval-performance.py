#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('metrics', Path(__file__).with_name('summarize-pi-humaneval-performance.py'))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class Reconciliation(unittest.TestCase):
    def test_weighted_clocks_and_incomplete_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {'task_id': 'HumanEval/156', 'start_utc': '2026-10-25T00:00:00Z', 'end_utc': '2026-10-25T00:01:00Z',
                    'pi': {'requests': [{'usage': {'input': 10, 'output': 3}}, {'usage': {'input': 20, 'output': 1}}]}}
            (root / 'tasks.jsonl').write_text(json.dumps(task) + '\n')
            clocks = [{'requestId': str(i), 'build': '0.3.0+' + 'a'*40, 'inputTokens': n, 'outputTokens': out,
                       'prefillMs': ms, 'decodeIntervals': max(0, out-1), 'decodeMs': decode,
                       'completedAt': '2026-10-25T00:00:30Z', 'executionPath': 'cuda-metal-split', 'kv': 'q8'}
                      for i, n, out, ms, decode in [(1, 10, 3, 10, 20), (2, 20, 1, 40, 0)]]
            log = root / 'worker.log'
            def save():
                log.write_text(''.join('INFERENCE_METRIC ' + json.dumps(c) + '\n' for c in clocks))
            save()
            result = metrics.summarize(root, log, 'a'*40)
            self.assertTrue(result['metrics_complete'])
            self.assertEqual(result['prefill_tokens_per_second'], 600)
            self.assertEqual(result['decode_tokens_per_second'], 100)
            clocks[0]['build'] = 'wrong'
            save()
            self.assertFalse(metrics.summarize(root, log, 'a'*40)['metrics_complete'])
            clocks.pop()
            save()
            result = metrics.summarize(root, log, 'a'*40)
            self.assertFalse(result['metrics_complete'])
            self.assertIsNone(result['prefill_tokens_per_second'])


if __name__ == '__main__':
    unittest.main()
