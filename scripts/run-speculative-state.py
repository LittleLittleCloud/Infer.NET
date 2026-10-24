#!/usr/bin/env python3
"""Runs the synthetic speculative-state correctness harness (Issue #72, S0) with a single model load.

Input is saved token IDs (never model output text) in one of two forms:
  --token-ids FILE            a JSON token-id array, or {"cases":[{"name","promptTokenIds","outputTokenIds"?}]}
  --humaneval-jsonl FILE      JSONL with a "prompt" field; tokenized once with `deepinfer qwen-tokenize`
With no matrix options the harness runs its practical default matrix (every reject offset at 128 tokens, accept-all/
mixed/reject-all at 4096, every token cap 1..block+2, natural end-token stops; warm 1 / formal 3 for the full-length
groups). Passing --output-lengths/--patterns/--blocks replaces it with their cross product (--ignore-eos allowed then).
Everything runs in ONE `speculative-check` process, so the 35B model is loaded once and the native graph uses the
deployed fused-fp32 decode, managed prefill chunk 32 and prepared scratch 128 unless overridden. The harness writes speculative-check.json (and failure-N.json evidence).
This is a state-correctness reference only: no draft model, no speedup or quality claim is produced.
"""
import argparse, json, pathlib, subprocess, sys, tempfile

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--exe', required=True, type=pathlib.Path, help='deepinfer executable (Mac build)')
parser.add_argument('--weights', default='models/qwen3.6-35b-a3b-8bit-mlx')
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument('--token-ids', type=pathlib.Path)
source.add_argument('--humaneval-jsonl', type=pathlib.Path)
parser.add_argument('--limit', type=int, default=3, help='maximum HumanEval prompts to tokenize')
parser.add_argument('--output-lengths', help='custom matrix: comma-separated token counts')
parser.add_argument('--patterns', help='custom matrix: comma-separated patterns')
parser.add_argument('--blocks', help='custom matrix: comma-separated block sizes 1..8')
parser.add_argument('--decode-attention', default='fused-fp32', choices=['fused-fp32', 'legacy'])
parser.add_argument('--metal-scratch-tokens', default='128', choices=['128', '512'])
parser.add_argument('--warm-runs', type=int, default=1)
parser.add_argument('--formal-runs', type=int, default=3)
parser.add_argument('--checkpoint-tokens', type=int, default=64)
parser.add_argument('--ignore-eos', action='store_true', help='decode a fixed number of tokens even after EOS')
parser.add_argument('--poison-suffix', action='store_true', help='fill rejected KV suffix rows with NaN at every checkpoint')
parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('artifacts/speculative-check'))
parser.add_argument('--timeout', type=int, default=24 * 3600)
args = parser.parse_args()

exe = str(args.exe.resolve())
work = pathlib.Path(tempfile.mkdtemp(prefix='speculative-ids-'))
try:
    if args.humaneval_jsonl:
        cases = []
        for number, line in enumerate(args.humaneval_jsonl.read_text(encoding='utf-8').splitlines()):
            if not line.strip():
                continue
            if len(cases) >= args.limit:
                break
            prompt_file = work / f'prompt-{number}.txt'
            prompt_file.write_text(json.loads(line)['prompt'], encoding='utf-8')
            done = subprocess.run([exe, 'qwen-tokenize', '--prompt-file', str(prompt_file), '--tokenizer', str(pathlib.Path(args.weights) / 'tokenizer.json')],
                                  capture_output=True, text=True, timeout=300)
            if done.returncode != 0:
                sys.exit('tokenize failed: ' + done.stderr[-500:])
            cases.append({'name': f'humaneval-{number}', 'promptTokenIds': json.loads(done.stdout)})
        ids = work / 'cases.json'
        ids.write_text(json.dumps({'cases': cases}))
    else:
        ids = args.token_ids
    command = [exe, 'speculative-check', '--weights', args.weights, '--token-ids', str(ids), '--output', str(args.output),
               '--decode-attention', args.decode_attention, '--metal-scratch-tokens', args.metal_scratch_tokens,
               '--warm-runs', str(args.warm_runs), '--formal-runs', str(args.formal_runs),
               '--checkpoint-tokens', str(args.checkpoint_tokens)]
    for name in ('output_lengths', 'patterns', 'blocks'):
        if getattr(args, name):
            command += ['--' + name.replace('_', '-'), getattr(args, name)]
    command += (['--ignore-eos'] if args.ignore_eos else []) + (['--poison-suffix'] if args.poison_suffix else [])
    print('RUN', ' '.join(command), flush=True)
    sys.exit(subprocess.run(command, timeout=args.timeout).returncode)
finally:
    for item in work.iterdir():
        item.unlink()
    work.rmdir()
