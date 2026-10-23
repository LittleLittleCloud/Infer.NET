#!/usr/bin/env python3
"""Independent FP32 oracle for the fixed 128-token split-prefix diagnostic.

MLX is used only to check the .NET/CUDA/Metal implementation, never to serve
inference or supply performance measurements. This script leaves the system's
wired-memory limit unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache


class Bf16StoredFp32AttentionCache(KVCache):
    """Round only persistent K/V to BF16; run attention on FP32 expansions."""

    def update_and_fetch(self, keys, values):
        stored = super().update_and_fetch(
            keys.astype(mx.bfloat16), values.astype(mx.bfloat16))
        return tuple(value.astype(mx.float32) for value in stored)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('serial', 'batch'), default='serial',
                        help='Serial is the validated oracle path; batch is diagnostic only')
    parser.add_argument('--kv-storage', choices=('fp32', 'bf16'), default='fp32',
                        help='BF16 stores K/V rounded to BF16 but expands them for FP32 attention; diagnostic only')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)

    ids = [100 + (i * 137) % 247000 for i in range(128)]
    model, _ = load(str(args.model), trust_remote_code=False)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    lm = model.language_model
    cache = model.make_cache()
    if args.kv_storage == 'bf16':
        cache = [Bf16StoredFp32AttentionCache() if isinstance(item, KVCache) else item
                 for item in cache]
    layers = []
    if args.mode == 'serial':
        traces = [np.empty((128, 2048), dtype=np.float32) for _ in range(20)]
        for position, token in enumerate(ids):
            x = lm.model.embed_tokens(mx.array([[token]]))
            for layer, block in enumerate(lm.model.layers):
                x = block(x, cache=cache[layer])
                if layer < 20:
                    mx.eval(x)
                    traces[layer][position] = np.asarray(x[0, -1], dtype=np.float32)
            # Materialize the whole model and its caches before the next token.
            logits = lm.lm_head(lm.model.norm(x))
            mx.eval(logits)
            if position % 16 == 15:
                print(f'Reference position={position + 1}/128', flush=True)
        for layer, data in enumerate(traces):
            if not np.isfinite(data).all():
                raise ValueError(f'Nonfinite layer {layer} output')
            raw = data.tobytes()
            (args.output / f'layer-{layer}.f32').write_bytes(raw)
            layers.append(hashlib.sha256(raw).hexdigest())
    else:
        x = lm.model.embed_tokens(mx.array([ids]))
        for layer, block in enumerate(lm.model.layers):
            x = block(x, cache=cache[layer])
            if layer < 20:
                mx.eval(x)
                data = np.asarray(x[0], dtype=np.float32)
                if data.shape != (128, 2048) or not np.isfinite(data).all():
                    raise ValueError(f'Unexpected layer {layer} output: {data.shape}')
                raw = data.tobytes()
                (args.output / f'layer-{layer}.f32').write_bytes(raw)
                layers.append(hashlib.sha256(raw).hexdigest())
        logits = lm.lm_head(lm.model.norm(x))
        mx.eval(logits)
    data = np.asarray(logits[0, -1], dtype=np.float32)
    if data.shape != (248320,) or not np.isfinite(data).all():
        raise ValueError(f'Unexpected logits: {data.shape}')
    raw = data.tobytes()
    (args.output / 'logits.f32').write_bytes(raw)
    source = json.loads((args.model / 'provenance.json').read_text())
    metadata = {
        'kind': ('independent-fp32-prefix-oracle-not-product-inference-or-benchmark'
                 if args.kv_storage == 'fp32' else
                 'independent-bf16-kv-fp32-attention-prefix-diagnostic-not-product-inference'),
        'modelRepository': source['repository'],
        'modelRevision': source['revision'],
        'weightBits': 8,
        'tokens': 128,
        'tokensSha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        'dtype': 'fp32',
        'batchTokens': 1 if args.mode == 'serial' else 128,
        'layerSha256': layers,
        'logitsSha256': hashlib.sha256(raw).hexdigest(),
        'argmax': int(np.argmax(data)),
        'limitations': ['Offline numerical oracle, not product inference',
                        'Batch mode has not passed the established FP32 control and must not be used for acceptance' if args.mode == 'batch' else 'Serial mode follows the established independent reference execution',
                        'Separate run; no network, state handoff, decode, or throughput measurement'],
    }
    if args.kv_storage == 'bf16':
        metadata['kvStorage'] = 'bf16'
        metadata['limitations'].append('BF16 K/V storage with FP32 attention arithmetic is a KV diagnostic, not the fixed FP32 oracle')
    (args.output / 'result.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'tokens': 128, 'layers': len(layers), 'argmax': metadata['argmax']}))


if __name__ == '__main__':
    main()
