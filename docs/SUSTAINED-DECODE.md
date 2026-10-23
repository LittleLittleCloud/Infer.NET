# 8K input / 4096 output sustained decode — 2026-10-23

The deployed affine8 model sustained a median **55.045 tokens/s** across three measured 4096-token generations after one warmup. The final 1024 tokens ran at **53.897 tokens/s** (97.96% of the overall rate), with zero swap-out in all four runs. Every 4096-token sequence matched; its first 128 token IDs also match the prior serial-prefix baseline.

| Trial | Prefill seconds | Client total seconds | Decode tok/s | Last 1024 tok/s |
|---|---:|---:|---:|---:|
| Warmup | 16.372 | 91.012 | 54.890 | 53.968 |
| 1 | 16.744 | 91.147 | 55.055 | 53.955 |
| 2 | 16.536 | 90.999 | 55.014 | 53.891 |
| 3 | 16.783 | 91.204 | 55.045 | 53.897 |

This is an **8K, FP32 GEMM/KV diagnostic**, not acceptance of the fixed BF16 / 131072-input / 4096-output targets. It establishes that the current pipeline's short-output decode rate is maintained across thousands of output tokens. A longer input changes attention cost and must be measured independently.

Configuration: pinned `mlx-community/Qwen3.6-35B-A3B-8bit` revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine8/group64, FP32 math/KV, CUDA prefix chunk32, Metal suffix128, send overlap on, greedy, ignoreEOS=true, no prefix-cache hit. One warmup plus three measured requests, exclusive GPU benchmark, AC power before and after. Same immutable source on both hosts: `2760e449364fbfd18f192d80aa66e41eb734a5da`. Network remains the existing home-router path.

Maximum response-sampled Metal allocation was 37,870,944,256 bytes (35.27 GiB); Windows whole-device peak was 19,585 MiB. Mac vmmap reported a 35.8G peak footprint. The benchmark checks the 38.5 GiB Metal, 22 GiB Windows and 512 MiB steady swap-out limits. Per-token timing windows are recorded; final-1024 rate sums the last 1024 measured decode intervals.

Public aggregate evidence: `site/long-output-summary.json`. Raw run `artifacts/dual-host/long-output-8192-4096-on-20261023T085635Z` is retained locally with per-token events, token IDs, host counters, power records, GPU sampling, stage logs and footprint. The summary contains the raw-summary/input hashes; host-specific raw files are not publicly downloadable. See [issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3) and [PR #19](https://github.com/LittleLittleCloud/Infer.NET/pull/19).

Reproduction uses the fixed 8192-token prompt from `artifacts/baselines/qwen36-context-fixed-data/8192-4096-prompt.txt` (SHA256 `defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`), streaming `/v1/infer/stream`, `maxTokens=4096`, `profile=true`, `benchmarkIgnoreEos=true`. The local orchestration and exact response evidence are retained in `artifacts/chat/benchmark-long-output.py` and the run directory. No framework substitution or lower weight precision was used.

## Larger input: 32768 / 4096, incremental KV

Source `788dd4a158cb6fee853f729b3f19032f3a774dbe`, CUDA32/Metal128, incremental KV, affine8/FP32 GEMM/KV, no cache. Warmup1 + measured3, AC, exclusive GPU workload. All four 4096-token sequences agree.

| Trial | Prefill s | Client total s | Decode tok/s | Last1024 tok/s |
|---|---:|---:|---:|---:|
| Warmup | 66.543 | 183.649 | 34.983 | 34.775 |
| 1 | 66.541 | 183.172 | 35.127 | 34.798 |
| 2 | 66.376 | 182.960 | 35.144 | 34.813 |
| 3 | 66.421 | 182.923 | 35.167 | 34.805 |

Medians: **66.421s prefill, 182.960s client total, 35.144tok/s decode, 34.805tok/s tail**. The rate is below the fixed goal number of37, even at32K; this is an FP32 diagnostic rather than the formal BF16/128K configuration. The input-dependent cost is material: the verified8K long-output rate was55.05tok/s. The experiment does not establish a transfer-mode regression, since32K final-handoff mode was not measured here.

Zero swap-out in all requests. Windows500ms sampling observed20115MiB maximum; response-sampled Metal allocation reached36.920GiB. These are observed samples, not continuous allocator high-water marks. No OOM or nondeterministic output occurred.

Public evidence: `site/long-context-summary.json`. Raw run `artifacts/dual-host/long-output-32768-4096-incremental-20261023T092422Z` remains local; its summary hash is published. Reproduce using the fixed32768-token file in the targets document, the prior HTTP options with maxTokens4096, and kv-transfer=incremental. Next work investigates the actual FP32 decode attention path and the unresolved BF16 numerical gate.
