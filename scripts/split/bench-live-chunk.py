#!/usr/bin/env python3
"""Measure an already-running authenticated split worker with a fixed input."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--endpoint", default="http://127.0.0.1:5201")
p.add_argument("--token-file")
p.add_argument("--config")
p.add_argument("--prompt-file", required=True)
p.add_argument("--chunk", type=int, choices=(32, 256, 512, 1024, 2048), required=True)
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--rounds", type=int, default=3)
p.add_argument("--max-tokens", type=int, default=8)
p.add_argument("--output", required=True)
a = p.parse_args()
prompt_bytes = Path(a.prompt_file).read_bytes()
prompt = prompt_bytes.decode()
token_path = a.token_file or (json.loads(Path(a.config).read_text())["token-file"] if a.config else None)
if token_path is None:
    p.error("--token-file or --config is required")
token = Path(token_path).read_text().strip()
payload = json.dumps({"prompt": prompt, "maxTokens": a.max_tokens,
                      "model": "qwen3.6-35b-a3b", "backend": "metal",
                      "profile": True, "benchmarkIgnoreEos": True}).encode()
rows = []
for index in range(a.warmup + a.rounds):
    request = urllib.request.Request(a.endpoint + "/v1/infer", data=payload,
        headers={"Content-Type": "application/json", "X-DeepInfer-Token": token})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    row = {"warmup": index < a.warmup, "clientMs": (time.monotonic()-started)*1000,
           "promptTokens": result["promptTokens"], "executionPath": result["executionPath"],
           "outputSha256": hashlib.sha256(json.dumps(result["tokenIds"]).encode()).hexdigest(),
           "timings": result["timings"]}
    rows.append(row)
    print(f"{'warmup' if row['warmup'] else 'formal'} {index}: "
          f"prefill={row['timings']['prefillMs']:.1f}ms client={row['clientMs']:.1f}ms "
          f"tokens={row['promptTokens']}", flush=True)
formal = rows[a.warmup:]
report = {"chunk": a.chunk, "warmup": a.warmup, "rounds": a.rounds,
          "promptSha256": hashlib.sha256(prompt_bytes).hexdigest(),
          "outputTokens": a.max_tokens, "rows": rows,
          "medianPrefillMs": statistics.median(r["timings"]["prefillMs"] for r in formal),
          "deterministic": len({r["outputSha256"] for r in rows}) == 1}
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(f"median prefill {report['medianPrefillMs']:.1f}ms; deterministic={report['deterministic']}")
