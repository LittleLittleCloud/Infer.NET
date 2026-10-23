# FP32 decode attention — 2026-10-23

A single-query Metal kernel reuses each KV group across its eight query heads. Four SIMD groups split key columns during QK, then feature columns during PV. It computes FP32 scores and accumulation in128-key partitions, without padding three unused query rows. The explicit policy is `decode-attention=fused-fp32`; CLI default remains legacy. No quantization, model, wire or CUDA arithmetic change.

## Same-package whole-model A/B

Both hosts: `3fee48879cd5b92c8525a0910fdc04a5b61095f5`. Pinned affine8/group64 weights, FP32 GEMM/KV, CUDA32/Metal128, incremental KV, CUDA send overlap on, zero prefix-cache hit, existing home-router LAN, AC. One excluded warmup and three measured trials per mode, with exclusive GPU workload. All4096 output IDs match across all eight32K requests; all128 IDs agree across all eight8K requests.

| Input / output | Mode | Prefill median s | Client median s | Decode median tok/s |
|---|---|---:|---:|---:|
| 32768 / 4096 | legacy | 66.438 | 183.082 | 35.096 |
| 32768 / 4096 | fused-fp32 | 66.553 | 156.765 | 45.431 |
| 8192 / 128 | legacy | 14.252 | 16.643 | 53.450 |
| 8192 / 128 | fused-fp32 | 14.452 | 16.596 | 59.445 |

32K decode improves **29.45%**, with client time reduced **14.37%**. The final1024-token median is45.241tok/s. Prefill is unchanged within observed variation; this optimization does not solve prefill. EightK short-output decode improves11.22%, but prefill dominates total request time.

Six chat cases, warm1 + measured3 each mode, have identical output sequences. Hello TTFT is91.011ms legacy and91.030ms fused. Short-context decode regresses by up to4.67% (explanation case); this tradeoff remains visible, and a context-dependent dispatch is next work. This is within the fixed5% regression number, but the configuration is diagnostic.

The final fused32K trial observed13.47MiB system swap-out; all other measured32K trials had zero. Each remained under512MiB. Windows500ms sampling observed20115MiB maximum. Metal allocation was sampled in responses and stayed below38.5GiB; this is not continuous allocator peak monitoring.

Both modes passed1748/1748 split reference checks on84 input/8 output, maximum relative L2=1.4991171297750514e-5 and maximum absolute error=.00012339651584625244. A standalone double-oracle test at17/2048/8192/32768 keys also passed; its hot-cache operator speedup is not substituted for whole-model throughput.

Live prefill cancellation recovered in1.619s with unchanged Mac/Windows model PIDs and the next8 output IDs matching the baseline. 15 deployment checks passed, including reset, streaming, cancellation, EOS, authentication and bounded queue behavior. An initial cancellation-test preflight was invalidated for stale Busy telemetry before any request was sent; the retry waited for idle health/telemetry.

## Remaining acceptance gap

These are FP32 /32K and8K results, **not formal BF16 /131072-input acceptance**. The fixed1600→2000tok/s prefill,37tok/s decode,200→180s total goals remain open in their specified configuration. Coordinator t2-t1 timing and continuous resource instrumentation remain to be completed. BF16 storage parity still fails its numerical gate; no tolerance or precision was silently changed.

Public evidence: `site/decode-attention-summary.json` includes every trial, hashes, input identity and raw run IDs. Raw SSE events, stage logs, token IDs and GPU/power records remain local under `artifacts/dual-host`; these paths are evidence identifiers, not public downloads. Reproduce with the fixed8K/32K inputs, `/v1/infer/stream`, profile=true, benchmarkIgnoreEos=true and the listed output lengths. Change only the explicit decode policy between paired cohorts.

See [PR21](https://github.com/LittleLittleCloud/Infer.NET/pull/21) and [issue3](https://github.com/LittleLittleCloud/Infer.NET/issues/3). Next: bounded Mac receive overlap, CUDA partitioned attention and BF16 arithmetic parity experiments.
