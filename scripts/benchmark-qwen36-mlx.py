#!/usr/bin/env python3
"""Single-process external MLX baseline; never used by the .NET inference path."""
import argparse
import base64
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SHORT_PROMPT = '请用中文详细解释什么是机器学习，并举出三个实际应用。'
DOCUMENT = ('机器学习通过数据学习规律，用于分类、回归和预测。训练集用于拟合模型，'
            '验证集用于选择超参数，测试集用于评估泛化能力。需要关注数据质量与过拟合。'
            'Distributed inference separates scheduling, model execution, and device memory. '
            'Each request owns its attention cache. Benchmark prefill and decode separately.\n')


def command(*args):
    result = subprocess.run(args, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path,
                        default=ROOT / 'models/qwen3.6-35b-a3b-5bit-mlx')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/baselines/qwen36-mlx-5bit.json')
    parser.add_argument('--prompt-tokens', type=int, nargs='+', default=[0, 4096, 16384],
                        help='0 uses the natural short chat prompt; other values are exact token counts')
    parser.add_argument('--prompt-file',type=Path,help='Use one exact saved UTF-8 prompt; requires one matching prompt-tokens case')
    parser.add_argument('--generation-tokens', type=int, default=256)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16',
                        help='Default checkpoint BF16 floats, or independent FP32 comparison at our precision')
    args = parser.parse_args()
    if args.trials < 1 or args.warmup < 1 or args.generation_tokens < 2:
        parser.error('Use at least one trial/warmup and two generated tokens')
    if any(n != 0 and n < 256 for n in args.prompt_tokens):
        parser.error('Exact-length fixtures require at least 256 tokens')

    if args.prompt_file and len(args.prompt_tokens)!=1:parser.error('--prompt-file requires one case')
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    provenance = json.loads((args.model / 'provenance.json').read_text())
    config = json.loads((args.model / 'config.json').read_text())
    display_info = json.loads(command('system_profiler', 'SPDisplaysDataType', '-json'))
    # Version strings alone cannot identify a locally edited baseline runtime.
    installed = importlib.metadata.distribution('mlx-lm')
    module_hashes = {}
    wanted = {'mlx_lm/generate.py', 'mlx_lm/models/base.py',
              'mlx_lm/models/gated_delta.py', 'mlx_lm/models/qwen3_next.py',
              'mlx_lm/models/qwen3_5.py', 'mlx_lm/models/qwen3_6.py'}
    for entry in installed.files or []:
        if str(entry) not in wanted:
            continue
        data = installed.locate_file(entry).read_bytes()
        digest = hashlib.sha256(data).digest()
        record_hash = entry.hash
        module_hashes[str(entry)] = dict(sha256=digest.hex(),
            matchesInstalledRecord=(base64.urlsafe_b64encode(digest).decode().rstrip('=') == record_hash.value)
                if record_hash and record_hash.mode == 'sha256' else None)
    report = {
        'startedAtUtc': datetime.now(timezone.utc).isoformat(),
        'engine': 'External MLX-LM, not DeepInfer.NET',
        'repository': provenance['repository'], 'revision': provenance['revision'],
        'modelDirectory': str(args.model.resolve()),
        'provenanceSha256': hashlib.sha256((args.model / 'provenance.json').read_bytes()).hexdigest(),
        'python': platform.python_version(), 'macOS': platform.mac_ver()[0],
        'machine': platform.machine(), 'device': mx.device_info(),
        'displayDevices': [{k: v for k, v in d.items() if k in
                           ['_name', 'sppci_model', 'sppci_cores']}
                          for d in display_info['SPDisplaysDataType']],
        'cpu': command('sysctl', '-n', 'machdep.cpu.brand_string'),
        'powerBefore': command('pmset', '-g', 'batt'),
        'thermalBefore': command('pmset', '-g', 'therm'),
        'swapBefore': command('sysctl', 'vm.swapusage'),
        'powerSettings': command('pmset', '-g', 'custom'),
        'benchmarkScriptSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'versions': {p: importlib.metadata.version(p) for p in
                     ['mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'huggingface-hub']},
        'runtimeModuleHashes': module_hashes,
        'runtimeEnvironment': {'MLX_GDN_PACKED': os.environ.get('MLX_GDN_PACKED', '1 (default)')},
        'batchSize': 1, 'trials': args.trials, 'warmupPerCase': args.warmup,
        'requestedGenerationTokens': args.generation_tokens,
        'prefillStepSize': 2048, 'sampling': 'greedy', 'thinking': False,
        'floatPrecision':args.precision,
        'prefixCache': False, 'speculativeDecode': False, 'kvQuantization': False,
        'forceOutputLength': True,
        'fixture': 'Tokenized repeated Chinese/English document; preserve final 128 chat tokens and truncate middle to exact length.',
        'decodeTiming': 'N-1 intervals, wall time from first emitted token through last; excludes model load, prefill and first token.',
        'promptTiming': 'MLX prompt_tokens / prompt_tps; includes producing the first token, per MLX implementation.',
        'quantization': config.get('quantization'), 'rows': [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    print('Loading external MLX baseline model…', flush=True)
    started = time.perf_counter()
    model, tokenizer = load(str(args.model), trust_remote_code=False)
    if args.precision == 'fp32':
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
    mx.synchronize()
    report['loadSeconds'] = time.perf_counter() - started
    print(f'Loaded in {report["loadSeconds"]:.2f}s; active MLX memory {mx.get_active_memory()/1e9:.2f} GB', flush=True)
    sampler = make_sampler(temp=0)

    def chat(text):
        return tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                                            tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False)

    # Natural EOS termination is retained for the functional sample.
    sample_prompt = chat('请用中文回答：17乘以23等于多少？再用一句话解释什么是机器学习。')
    sample = list(stream_generate(model, tokenizer, sample_prompt, max_tokens=128,
                                 sampler=sampler, prefill_step_size=2048))
    answer = ''.join(s.text for s in sample)
    report['functionalSample'] = {'promptTokenIds': sample_prompt,
                                  'outputTokenIds': [s.token for s in sample],
                                  'text': answer, 'finishReason': sample[-1].finish_reason}
    print('Functional sample:', answer, flush=True)
    if '391' not in answer:
        save()
        raise RuntimeError('Functional sample did not contain the expected arithmetic answer 391')
    save()

    def run(prompt):
        gc.collect()
        mx.clear_cache()
        mx.reset_peak_memory()
        mx.random.seed(0)
        ids, text, times = [], [], []
        original_eos = tokenizer._eos_token_ids
        tokenizer._eos_token_ids = set()  # Same fixed-output policy as mlx_lm.benchmark.
        started = time.perf_counter()
        try:
            for response in stream_generate(model, tokenizer, prompt,
                                             max_tokens=args.generation_tokens,
                                             sampler=sampler, prefill_step_size=2048):
                times.append(time.perf_counter())
                ids.append(response.token)
                text.append(response.text)
        finally:
            tokenizer._eos_token_ids = original_eos
        mx.synchronize()
        ended = time.perf_counter()
        if len(ids) != args.generation_tokens:
            raise RuntimeError(f'Expected fixed output length, got {len(ids)}')
        intervals = len(ids) - 1
        decode = times[-1] - times[0]
        return {'promptTokens': len(prompt), 'generatedTokens': len(ids),
                'ttftMs': (times[0] - started) * 1000,
                'decodeIntervals': intervals, 'decodeMs': decode * 1000,
                'decodeTokensPerSecond': intervals / decode,
                'endToEndMs': (ended - started) * 1000,
                'endToEndTokensPerSecond': len(ids) / (ended - started),
                'mlxPromptMs': len(prompt) / response.prompt_tps * 1000,
                'mlxPromptTokensPerSecond': response.prompt_tps,
                'mlxReportedGenerationTokensPerSecond': response.generation_tps,
                'peakMlxMemoryGB': mx.get_peak_memory() / 1e9,
                'finishReason': response.finish_reason,
                'outputTokenIds': ids, 'outputText': ''.join(text),
                'emissionOffsetsMs': [(t - started) * 1000 for t in times]}

    for target in args.prompt_tokens:
        if args.prompt_file:
            prompt=chat(args.prompt_file.read_text())
            if len(prompt)!=target:raise ValueError(f'Exact prompt has {len(prompt)} tokens, expected {target}')
        elif target == 0:
            prompt = chat(SHORT_PROMPT)
        else:
            raw = chat(DOCUMENT * (target // 40 + 10) + '\n' + SHORT_PROMPT)
            if len(raw) < target:
                raise RuntimeError('Document fixture is shorter than its target')
            prompt = raw[:target - 128] + raw[-128:]
        print(f'Case: {len(prompt)} prompt tokens, {args.generation_tokens} output tokens', flush=True)
        row = {'case': 'short-chat' if target == 0 else f'{target}-token-document',
               'promptTokens': len(prompt), 'promptTokenIds': prompt,
               'promptTokenIdsSha256':hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest(),
               'promptFile':str(args.prompt_file.resolve()) if args.prompt_file else None,
               'promptSha256':hashlib.sha256(args.prompt_file.read_bytes()).hexdigest() if args.prompt_file else None,
               'warmups': [], 'samples': []}
        report['rows'].append(row)
        for index in range(args.warmup):
            result = run(prompt)
            row['warmups'].append(result)
            print(f'  warmup {index+1}: {result["decodeTokensPerSecond"]:.2f} token/s (excluded)', flush=True)
            save()
        for index in range(args.trials):
            result = run(prompt)
            row['samples'].append(result)
            print(f'  trial {index+1}: prefill={result["mlxPromptTokensPerSecond"]:.1f} token/s; '
                  f'TTFT={result["ttftMs"]:.1f}ms; decode={result["decodeTokensPerSecond"]:.2f} token/s; '
                  f'peak={result["peakMlxMemoryGB"]:.2f}GB', flush=True)
            save()
        samples = row['samples']
        row['summary'] = {
            'decodeTokensPerSecond': sum(s['decodeIntervals'] for s in samples) * 1000 / sum(s['decodeMs'] for s in samples),
            'meanDecodeTokensPerSecond': statistics.mean(s['decodeTokensPerSecond'] for s in samples),
            'stdevDecodeTokensPerSecond': statistics.stdev(s['decodeTokensPerSecond'] for s in samples) if len(samples)>1 else 0,
            'promptTokensPerSecond': sum(s['promptTokens'] for s in samples) * 1000 / sum(s['mlxPromptMs'] for s in samples),
            'medianTtftMs': statistics.median(s['ttftMs'] for s in samples),
            'peakMlxMemoryGB': max(s['peakMlxMemoryGB'] for s in samples),
            'outputIdsIdentical': all(s['outputTokenIds'] == samples[0]['outputTokenIds'] for s in samples),
        }
        save()
    report['completedAtUtc'] = datetime.now(timezone.utc).isoformat()
    report['powerAfter'] = command('pmset', '-g', 'batt')
    report['thermalAfter'] = command('pmset', '-g', 'therm')
    report['swapAfter'] = command('sysctl', 'vm.swapusage')
    save()
    print('Baseline evidence:', args.output, flush=True)


if __name__ == '__main__':
    main()
