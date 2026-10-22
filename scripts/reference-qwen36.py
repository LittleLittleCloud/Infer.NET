#!/usr/bin/env python3
"""Independent verification oracle ONLY. Not used by the .NET inference/benchmark path."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import maybe_set_recommended_wired_limit

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--model', type=Path, default=ROOT / 'models/qwen3.6-35b-a3b-5bit-mlx')
p.add_argument('--output', type=Path, default=ROOT / 'artifacts/reference/qwen36')
args = p.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
directory = args.model
# Match MLX-LM's normal generation residency policy. Manual layer-by-layer
# verification otherwise leaves large 8-bit weights needlessly pageable.
previous_wired_limit = maybe_set_recommended_wired_limit()
model, tokenizer = load(str(directory), trust_remote_code=False)
# Quantized integer weights remain packed; floats/scales and all activations use
# FP32 to compare the original .NET/Metal implementation at the same precision.
model.set_dtype(mx.float32)
mx.eval(model.parameters())
prompt = '用一句话介绍你自己。'
ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                     tokenize=True, add_generation_prompt=True,
                                     enable_thinking=False)
inputs = list(ids)
trace_positions = {0, len(ids)-1, len(ids)+6}
cache = model.make_cache()
generated = []
position = 0
started = time.monotonic()
lm = model.language_model
while position < len(inputs):
    x = lm.model.embed_tokens(mx.array([[inputs[position]]]))
    for layer, block in enumerate(lm.model.layers):
        x = block(x, cache=cache[layer])
        if position in trace_positions:
            mx.eval(x)
            np.asarray(x, dtype=np.float32).tofile(OUT / f'p{position}-layer{layer}.f32')
    logits = lm.lm_head(lm.model.norm(x))
    mx.eval(logits)
    if position >= len(ids)-1:
        token = int(mx.argmax(logits[0, -1]).item())
        generated.append(token)
        if len(generated) < 8:
            inputs.append(token)
    position += 1
    print(f'Reference position={position}/{len(inputs)} elapsed={time.monotonic()-started:.2f}s activeGB={mx.get_active_memory()/1e9:.3f}', flush=True)
np.asarray(logits[0, -1], dtype=np.float32).tofile(OUT / 'logits.f32')
(OUT / 'reference.json').write_text(json.dumps({
    'purpose': 'Independent FP32 numerical oracle; not the product inference engine',
    'modelDirectory': str(directory.resolve()),
    'provenance': json.loads((directory / 'provenance.json').read_text()),
    'previousWiredLimit': previous_wired_limit,
    'device': mx.device_info(),
    'prompt': prompt, 'promptTokenIds': ids, 'inputTokenIds': inputs,
    'generatedTokenIds': generated, 'text': tokenizer.decode(generated),
    'tracePositions': sorted(trace_positions),
}, ensure_ascii=False, indent=2)+'\n')
print('Independent reference:',tokenizer.decode(generated),flush=True)
print('Reference artifacts:', OUT, flush=True)
