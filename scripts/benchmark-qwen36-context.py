#!/usr/bin/env python3
"""Plan or measure long-context RPC workloads. Tokenizers is used only to prepare data, never for inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'qwen3.6-35b-a3b'
DEFAULT_CASES = ['128:256', '2048:4096', '8192:4096', '32768:8192',
                 '131072:8192', '245760:16384', '270000:8192']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def planned_capacity(total):
    return max(2048, ((total + 2047) // 2048) * 2048)


def expected_work(prompt, outputs, chunk=2048, attention="fused", experts="grouped", delta="chunked", gemm="fp32", gemm_impl="staged", attention_impl="fp32", expert_impl="staged", attention_batch=128, expert_flow="staged"):
    sizes = [min(chunk, prompt-start) for start in range(0, prompt, chunk)]
    long_tiles = sum(min(prompt, start+chunk, start+tile+attention_batch) > 2048 for start in range(0,prompt,chunk)
                     for tile in range(0,min(chunk,prompt-start),attention_batch)) if attention == "legacy" else 0
    attention_tiles = sum((n+attention_batch-1)//attention_batch-1 for n in sizes)
    long_decode = outputs-1 if attention_impl=="bf16-nax" else max(0, prompt + outputs - 1 - max(prompt, 2048))
    grouped = sum(n>=64 for n in sizes) if experts == "grouped" else 0
    expansion = (860 if gemm_impl=="expanded" else 660)*sum(n>=64 for n in sizes) if gemm=="bf16-nax" and gemm_impl!="staged" else 0
    if expert_impl=="packed" and gemm=="bf16-nax":expansion+=80*sum(n>=64 for n in sizes)
    if expert_flow=="direct": expansion-=40*sum(n>=64 for n in sizes)
    packing = 10*sum((n+attention_batch-1)//attention_batch for n in sizes) if attention_impl=="bf16-nax" and attention!="legacy" else 0
    delta_chunks = sum(n>=64 for n in sizes) if delta == "chunked" else 0
    return (len(sizes)+outputs-1,
            len(sizes)*672+3+(outputs-1)*675+10*(long_tiles+long_decode)
            +20*attention_tiles+80*grouped+30*delta_chunks+expansion+packing)


def snapshot():
    result = {'elapsedClock': time.monotonic(), 'loadAverage': os.getloadavg()}
    for name, command in [('swap', ['sysctl', 'vm.swapusage']), ('vm', ['vm_stat']), ('thermal', ['pmset', '-g', 'therm']), ('power', ['pmset', '-g', 'batt']), ('powerSettings', ['pmset', '-g', 'custom'])]:
        run = subprocess.run(command, capture_output=True, text=True)
        result[name] = run.stdout.strip() if run.returncode == 0 else None
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-directory', type=Path, default=ROOT/'models/qwen3.6-35b-a3b-5bit-mlx')
    p.add_argument('--endpoint', default='http://127.0.0.1:5100')
    p.add_argument('--host', type=Path, default=ROOT/'artifacts/native/osx-arm64/deepinfer')
    p.add_argument('--corpus', type=Path, help='UTF-8 document/code corpus. Default: a recorded snapshot of this repository source/docs.')
    p.add_argument('--prompt-file', type=Path, help='Use one exact saved prompt unchanged; requires one matching input-token budget')
    p.add_argument('--chunk', type=int, choices=[32,64,128,256,512,1024,2048], default=2048)
    p.add_argument('--attention', choices=['legacy','fused'], default='fused')
    p.add_argument('--experts', choices=['vector','grouped'], default='grouped')
    p.add_argument('--gemm', choices=['fp32','bf16-nax'], default=os.environ.get('DEEPINFER_METAL_PREFILL_GEMM','fp32'))
    p.add_argument('--gemm-impl', choices=['staged','hybrid','expanded'], default=os.environ.get('DEEPINFER_METAL_PREFILL_GEMM_IMPL','staged'))
    p.add_argument('--expert-impl',choices=['staged','packed'],default=os.environ.get('DEEPINFER_METAL_PREFILL_EXPERT_IMPL','staged'))
    p.add_argument('--expert-flow',choices=['staged','direct'],default=os.environ.get('DEEPINFER_METAL_PREFILL_EXPERT_FLOW','staged'))
    p.add_argument('--expert-pipeline',choices=['serial','double-buffer'],default=os.environ.get('DEEPINFER_METAL_PREFILL_EXPERT_PIPELINE','serial'))
    p.add_argument('--attention-impl', choices=['fp32','bf16-nax'], default=os.environ.get('DEEPINFER_METAL_PREFILL_ATTENTION_IMPL','fp32'))
    p.add_argument('--kv', choices=['fp32','bf16'], default=os.environ.get('DEEPINFER_METAL_KV','fp32'))
    p.add_argument('--delta', choices=['sequential','chunked'], default='chunked')
    p.add_argument('--cases', nargs='+', default=DEFAULT_CASES, help='input:output token budgets, e.g. 8192:4096')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/baselines/qwen36-context.json')
    p.add_argument('--run', action='store_true', help='Execute supported cases; otherwise save a plan only')
    p.add_argument('--fixed-output', action='store_true', help='Explicit throughput diagnostic: continue past EOS, not a natural completion')
    p.add_argument('--warmup',type=int,default=0)
    p.add_argument('--key-tokens',type=int,choices=[32,64,128],default=int(os.environ.get('DEEPINFER_METAL_PREFILL_KEY_TILE','32')))
    p.add_argument('--attention-batch',type=int,choices=[128,256,512,1024,2048],default=int(os.environ.get('DEEPINFER_METAL_PREFILL_ATTENTION_BATCH','128')))
    p.add_argument('--delta-impl',choices=['staged','device'],default=os.environ.get('DEEPINFER_METAL_PREFILL_DELTA_IMPL','staged'))
    p.add_argument('--expert-gate',choices=['split','fused'],default=os.environ.get('DEEPINFER_METAL_PREFILL_EXPERT_GATE','split'))
    p.add_argument('--expert-tiles',choices=['fixed','adaptive'],default=os.environ.get('DEEPINFER_METAL_PREFILL_EXPERT_TILES','fixed'))
    p.add_argument('--attention-partition',type=int,choices=[1024,2048,4096,8192,16384,32768],default=int(os.environ.get('DEEPINFER_METAL_PREFILL_ATTENTION_PARTITION','1024')))
    p.add_argument('--query-tokens',type=int,choices=[4,8],default=4)
    p.add_argument('--trials', type=int, default=1)
    p.add_argument('--timeout', type=float, default=86400)
    p.add_argument('--worker-pid', type=int)
    p.add_argument('--require-ac', action='store_true', help='Reject requests without AC power before, during, or after measurement')
    a = p.parse_args()
    def cancel(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, cancel)
    if a.trials < 1 or a.warmup<0 or not 0 < a.timeout <= 86400:
        p.error('trials >= 1 and 0 < timeout <= 86400 required')
    cases = []
    for case in a.cases:
        if not re.fullmatch(r'[1-9][0-9]*:[1-9][0-9]*', case):
            p.error('Cases must be positive input:output token budgets')
        inp, out = map(int, case.split(':'))
        if not 32 <= inp <= 1048576 or not 2 <= out <= 32768:
            p.error('Input budget 32..1048576, output budget 2..32768 required')
        cases.append((inp, out))
    if a.prompt_file and (a.corpus or len(cases) != 1):
        p.error('--prompt-file requires one case and cannot be combined with --corpus')
    try:
        from tokenizers import Tokenizer
    except ImportError:
        p.error('Use artifacts/baselines/mlx-env/bin/python (tokenizers); MLX is not imported or used')
    tokenizer_path = a.model_directory/'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    def chat(text):
        return [248045] + enc('user\n') + enc(text) + [248046] + enc('\n') + [248045] + enc('assistant\n') + [248068] + enc('\n\n') + [248069] + enc('\n\n')
    model_config = json.loads((tokenizer_path.parent/'config.json').read_text())
    config = model_config['text_config']
    limit = config['max_position_embeddings']
    paths = [(a.prompt_file or a.corpus).resolve()] if (a.prompt_file or a.corpus) else sorted(
        [*ROOT.glob('src/**/*.cs'), *ROOT.glob('native/DeepInfer.Metal/*.mm'), *ROOT.glob('docs/*.md')])
    paths = [path for path in paths if 'obj' not in path.parts and 'bin' not in path.parts]
    sources = [{'path': str(path), 'sha256': sha(path.read_bytes())} for path in paths]
    corpus = '\n\n'.join(f'--- {path.name} ---\n{path.read_text()}' for path in paths)
    if not corpus.strip():
        p.error('Corpus is empty')
    report = {'startedAtUtc': datetime.now(timezone.utc).isoformat(), 'mode': 'fixed-output-throughput-EOS-ignored' if a.fixed_output else 'natural-EOS',
              'engine': 'DeepInfer.NET Native AOT + original Metal kernels', 'macOS': platform.mac_ver()[0],
              'nativeContextLimit': limit, 'contextIncludesInputAndOutput': True, 'concurrency': 1,
              'precision': {'weights': model_config['quantization'], 'activationBuffers': 'fp32', 'state': 'fp32',
                  'prefillGemm': a.gemm,
                  'kv': a.kv, 'gemmImpl': a.gemm_impl, 'attentionImpl': a.attention_impl, 'expertImpl':a.expert_impl},
              'modelDirectory': str(a.model_directory.resolve()), 'provenanceSha256': sha((a.model_directory/'provenance.json').read_bytes()),
              'prefixReuse': False, 'prefillChunkTokens': a.chunk, 'prefillAttention': a.attention, 'prefillExperts': a.experts, 'prefillDelta': a.delta, 'expertFlow': a.expert_flow, 'expertTiles': a.expert_tiles, 'expertGate': a.expert_gate, 'deltaImpl': a.delta_impl, 'attentionPartitionTokens': a.attention_partition,
              'attentionQueryTileTokens': a.attention_batch, 'attentionComputeQueryTokens': a.query_tokens, 'attentionKeyTileTokens':a.key_tokens, 'deltaBlockTokens': 32,
              'attentionFeatureTile': os.environ.get('DEEPINFER_METAL_PREFILL_ATTENTION_FEATURES', 'default'),
              'excludedWarmupRequests': a.warmup, 'modelLoadingExcluded': True, 'requireAcPower': a.require_ac, 'expertPipeline': a.expert_pipeline,
              'capacityAllocationIncludedInFirstToken': True, 'sources': sources, 'cases': [],
              'scriptSha256': sha(Path(__file__).read_bytes()), 'tokenizerSha256': sha(tokenizer_path.read_bytes())}
    if a.host.exists():
        report['nativeExecutableSha256'] = sha(a.host.read_bytes())
        report['nativeMetalLibrarySha256'] = sha((a.host.parent/'libdeepinfer_metal.dylib').read_bytes())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    data_dir = a.output.parent/(a.output.stem+'-data')
    data_dir.mkdir(exist_ok=True)
    def save():
        temp = a.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        temp.replace(a.output)
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('DEEPINFER_TOKEN'):
        headers['X-DeepInfer-Token'] = os.environ['DEEPINFER_TOKEN']
    for budget, output in cases:
        row = {'inputBudget': budget, 'requestedOutputTokens': output, 'requestedTotalTokens': budget+output}
        report['cases'].append(row)
        if budget + output > limit:
            row.update(status='unsupported', reason=f'Total exceeds pinned unscaled RoPE limit {limit}; no YaRN implemented')
            print(json.dumps(row), flush=True); save(); continue
        if a.prompt_file:
            prompt = a.prompt_file.read_text(); ids = chat(prompt)
            if len(ids) != budget:
                p.error(f'Exact prompt has {len(ids)} tokens; input budget must match')
            repeats = 1; lo = len(corpus)
        else:
            prefix = '请审阅以下项目材料，给出详细的架构、正确性、性能和测试分析，逐项引用材料中的具体事实。\n\n材料：\n'
            suffix = '\n\n请输出完整审阅报告，逐模块展开，给出具体依据、风险和改进步骤。不要用简短摘要替代分析。'
            minimum = len(chat(prefix+suffix))
            if budget < minimum:
                p.error(f'Input budget must include at least {minimum} instruction/template tokens')
            repeats = max(1, math.ceil(budget / max(1, len(enc(corpus)))))
            material = corpus * repeats
            while len(chat(prefix+material+suffix)) < budget:
                material += corpus; repeats += 1
            lo, hi = 0, len(material)
            while lo < hi:
                mid = (lo + hi + 1)//2
                if len(chat(prefix+material[:mid]+suffix)) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            prompt = prefix+material[:lo]+suffix
            ids = chat(prompt)
        row.update(inputTokens=len(ids), corpusCopiesPrepared=repeats, corpusRepeatsUsed=math.ceil(lo/len(corpus)),
                   promptSha256=sha(prompt.encode()), promptTokenIdsSha256=sha(json.dumps(ids, separators=(',', ':')).encode()),
                   plannedCapacity=planned_capacity(len(ids)+output),
                   plannedFp32KvGiB=planned_capacity(len(ids)+output)*40960/2**30,
                   plannedKvGiB=planned_capacity(len(ids)+output)*(20480 if a.kv=='bf16' else 40960)/2**30)
        prompt_path = data_dir/f'{budget}-{output}-prompt.txt'
        prompt_path.write_text(prompt)
        row['promptFile'] = str(prompt_path)
        # Own tokenizer is authoritative. No GPU/model loading in this command.
        own = subprocess.run([str(a.host.resolve()), 'qwen-tokenize', '--model', MODEL,
                              '--prompt-file', str(prompt_path.resolve()), '--tokenizer', str(tokenizer_path.resolve())], cwd=ROOT, capture_output=True, text=True, check=True)
        if json.loads(own.stdout) != ids:
            raise ValueError('Own C# and preparation tokenizer IDs differ')
        row['ownTokenizerVerified'] = True
        if not a.run:
            row['status'] = 'planned'; print(json.dumps(row), flush=True); save(); continue
        row.update(status='running', trials=[], memoryBefore=snapshot())
        save(); print(f'Starting input={len(ids)} requestedOutput={output} mode={report["mode"]}', flush=True)
        for iteration in range(a.trials+a.warmup):
            trial=iteration-a.warmup
            label=f'warmup{iteration+1}' if trial<0 else f'trial{trial+1}'
            target=row.setdefault('warmups',[]) if trial<0 else row['trials']
            power_before = snapshot()
            if a.require_ac and "Now drawing from 'AC Power'" not in (power_before['power'] or ''):
                row.update(status='failed', error='AC power required before request', powerBefore=power_before)
                save()
                raise RuntimeError(row['error'])
            samples, done = [], threading.Event()
            memory_path = data_dir/f'{budget}-{output}-{label}-memory.jsonl'
            row['memorySamplesFile'] = str(memory_path); save()
            def sample():
                with memory_path.open('w', buffering=1) as memory_log:
                    while not done.wait(5):
                        s = snapshot()
                        if a.worker_pid:
                            run = subprocess.run(['ps', '-o', 'rss=', '-p', str(a.worker_pid)], capture_output=True, text=True)
                            s['workerRssKiB'] = int(run.stdout.strip()) if run.stdout.strip() else None
                        samples.append(s)
                        memory_log.write(json.dumps(s)+'\n')
            monitor = threading.Thread(target=sample, daemon=True); monitor.start()
            started = time.perf_counter()
            try:
                payload = dict(prompt=prompt, maxTokens=output, model=MODEL, backend='metal', profile=True, benchmarkIgnoreEos=a.fixed_output)
                request = urllib.request.Request(a.endpoint+'/v1/infer', json.dumps(payload, ensure_ascii=False).encode(), headers)
                with urllib.request.urlopen(request, timeout=a.timeout) as response:
                    result = json.load(response)
                rpc_ms = (time.perf_counter()-started)*1000
                power_after = snapshot()
                if a.require_ac and any("Now drawing from 'AC Power'" not in (s['power'] or '') for s in [power_before, *list(samples), power_after]):
                    raise RuntimeError('AC power lost during request')
                if result['isSimulation'] or result['model'] != MODEL or result['promptTokens'] != len(ids):
                    raise ValueError('Unexpected model/simulation/tokenizer result')
                count = len(result['tokenIds'])
                if a.fixed_output and (count != output or result['finishReason'] != 'length'):
                    raise ValueError('Fixed-length throughput run did not complete requested output')
                if count == 0:
                    raise ValueError('Model emitted no output')
                submissions, dispatches = expected_work(len(ids), count, a.chunk, a.attention, a.experts, a.delta, a.gemm, a.gemm_impl, a.attention_impl, a.expert_impl, a.attention_batch, a.expert_flow)
                # 8-bit prefill submits the final vocabulary projection separately
                # to bound the per-command resident resource set at long contexts.
                if report['precision']['weights']['bits'] == 8:
                    submissions += 1
                # Natural EOS requires a final classifier forward which emits no visible token.
                if result['finishReason'] == 'eos':
                    submissions += 1
                    dispatches += 675 + (10 if len(ids)+count-1 >= 2048 else 0)
                if result['gpuSubmissions'] != submissions or result['gpuDispatches'] != dispatches:
                    raise ValueError(f'Unexpected GPU work counts: {result["gpuDispatches"]}/{result["gpuSubmissions"]}, expected {dispatches}/{submissions}')
                raw = data_dir/f'{budget}-{output}-{label}.json'
                raw.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
                t = result['timings']
                windows = t['decodeWindows']
                if sum(w['endOutputToken']-w['startOutputToken']+1 for w in windows) != count-1:
                    raise ValueError('Profile windows do not cover all decode intervals')
                if abs(sum(w['milliseconds'] for w in windows)-t['decodeMs']) > .001:
                    raise ValueError('Profile window elapsed times differ from aggregate decode time')
                if any(w['contextTokens'] != len(ids)+w['endOutputToken']-1 for w in windows):
                    raise ValueError('Profile window context length is incorrect')
                target.append({'rpcMs': rpc_ms, 'powerBefore': power_before, 'powerAfter': power_after, 'actualOutputTokens': count,
                    'finishReason': result['finishReason'], 'prefillTokensPerSecond': len(ids)*1000/t['prefillMs'],
                    'timings': t, 'rawResultFile': str(raw), 'rawResultSha256': sha(raw.read_bytes())})
                print(json.dumps(target[-1], ensure_ascii=False), flush=True)
            except BaseException as error:
                row.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error), elapsedSeconds=time.perf_counter()-started)
                if isinstance(error, urllib.error.HTTPError):
                    row['httpErrorBody'] = error.read().decode(errors='replace')
                raise
            finally:
                done.set(); monitor.join(timeout=6)
                row.setdefault('memorySamples', []).extend(samples)
                row['memoryAfter'] = snapshot(); save()
        row['status'] = 'completed'; save()
    report['completedAtUtc'] = datetime.now(timezone.utc).isoformat(); save()
    print(f'Report: {a.output}', flush=True)


if __name__ == '__main__':
    main()
