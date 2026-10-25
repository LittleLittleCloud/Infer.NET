#!/usr/bin/env python3
"""Independent MLX/CPU Q8 oracle only; never used as the product backend."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache
import numpy as np

spec = importlib.util.spec_from_file_location('oracle_codec', Path(__file__).with_name('q8-kv-fixtures.py'))
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


class ReferenceQ8Cache(KVCache):
    def __init__(self):
        super().__init__()
        self.packed = [bytearray(), bytearray()]

    def update_and_fetch(self, keys, values):
        restored = []
        for kind, tensor in enumerate((keys, values)):
            mx.eval(tensor)
            source = np.asarray(tensor, dtype=np.float32).transpose(0, 2, 1, 3)
            packed = codec.encode(source.reshape(-1).tolist())
            self.packed[kind].extend(packed)
            decoded = []
            for at in range(0, len(packed), 34):
                scale = struct.unpack_from('<e', packed, at)[0]
                decoded.extend(scale * v for v in struct.unpack_from('32b', packed, at+2))
            restored.append(mx.array(np.asarray(decoded, dtype=np.float32).reshape(source.shape).transpose(0, 2, 1, 3)))
        return super().update_and_fetch(*restored)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trace-all', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load(str(a.model), trust_remote_code=False)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    prompt = '用一句话介绍你自己。'
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False)
    inputs, generated = list(ids), []
    positions = {0, len(ids)-1, len(ids)+6}
    cache = model.make_cache()
    for i, c in enumerate(cache):
        if isinstance(c, KVCache):
            cache[i] = ReferenceQ8Cache()
    lm = model.language_model
    started = time.monotonic()
    position = 0
    while position < len(inputs):
        x = lm.model.embed_tokens(mx.array([[inputs[position]]]))
        for layer, block in enumerate(lm.model.layers):
            x = block(x, cache=cache[layer])
            if a.trace_all or position in positions:
                mx.eval(x)
                np.asarray(x, dtype=np.float32).tofile(a.output / f'p{position}-layer{layer}.f32')
        logits = lm.lm_head(lm.model.norm(x))
        mx.eval(logits)
        if position == len(ids)-1:
            for layer, c in enumerate(cache):
                if layer >= 20:
                    break
                if isinstance(c, ReferenceQ8Cache):
                    for kind in range(2):
                        (a.output / f'prompt-state-{layer}-{kind}.bin').write_bytes(c.packed[kind])
                else:
                    for native_kind, cache_index in ((0, 1), (1, 0)):
                        mx.eval(c[cache_index])
                        np.asarray(c[cache_index], dtype=np.float32).tofile(a.output / f'prompt-state-{layer}-{native_kind}.bin')
        if position >= len(ids)-1:
            token = int(mx.argmax(logits[0, -1]).item())
            generated.append(token)
            np.asarray(logits[0, -1], dtype=np.float32).tofile(a.output / f'logits-{len(generated)-1}.f32')
            if len(generated) < 8:
                inputs.append(token)
        position += 1
        print(json.dumps({'position': position, 'elapsed_seconds': time.monotonic()-started}), flush=True)
    np.asarray(logits[0, -1], dtype=np.float32).tofile(a.output / 'logits.f32')
    result = {'purpose': __doc__, 'prompt': prompt, 'promptTokenIds': ids, 'inputTokenIds': inputs,
              'generatedTokenIds': generated, 'tracePositions': sorted(positions),
              'kv': 'q8-block32-fp16-scale-rne-le', 'arithmetic': 'FP32; independent CPU Q8 conversion',
              'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'codec_script_sha256': hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest()}
    (a.output / 'reference.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
