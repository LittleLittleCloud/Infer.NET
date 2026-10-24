# CUDA Batch Prefill: 8K Dual-Machine A/B

2026-10-23, associated with Issue #3. Fixed Qwen3.6 35B A3B affine8/group64, revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`.

**8K input, 128 fixed output: worker prefill reduced from 52.503 seconds to 18.642 seconds, a 2.82 times improvement; client entire request reduced from 54.782 seconds to 20.926 seconds.** This is the deployment diagnostic path for FP32 KV / FP32 multiplication, and has not yet passed the formal threshold of BF16, 128K, or 4096 output.

## Changes and Numerical Verification

The original CUDA path executes all 20 layers per token. The new path executes batch projection, convolution, attention, routing, and expert calculations per layer within each 32-token block; DeltaNet keeps state rows in registers to complete the entire block recursion, reducing repeated reads/writes and kernel launches. Projection/experts still retain the original FP32 GEMV accumulation order. `cuda-prefill=serial|batch-fp32` can be switched with the same binary, defaulting to batch-fp32.

There are additionally self-implemented SM89 affine8 → BF16 Tensor Core GEMM primitives, **not included in this round of deployment performance data**. Independent BF16 operand comparison: 32/128/512-token large matrix is approximately 3.0/10.1/14.4 times; Delta primitive is approximately 7–8 times. These operator ratios cannot be taken as the complete model benefit.

Complete fixed prefix native test uses 131 synthetic valid token IDs, 1/7/17/32 non-aligned chunking, comparing all 20 layer traces, hidden states, 40 KV/Delta/convolution state tensors. FP32 and BF16 KV two tiers are both limited and byte-identical. 512-token prefix warmup 1 + formal 3 times: FP32 KV 2.628705 → 0.764376 seconds (3.44 times); BF16 KV 2.617131 → 0.765083 seconds (3.42 times). This is a comparison of two execution paths within the same CUDA, **does not solve the BF16 cross-host error threshold between CUDA and Metal**.

## Dual-machine measurement

Both ends use the same source `cc19a65056f351e201f65901d285490ee24b11b1`, Native AOT / CUDA ABI 5, Mac M5 Pro 20 cores / 48 GiB + Windows RTX 4090. Mac AC, `pmset powermode=2`; existing Mac Wi-Fi / Windows wired LAN, TLS data channel, no SSH tunnel. Only the Windows execution policy is switched; the Mac serve PID and packages remain unchanged. 32-token wire chunk, no prefix cache, reset each time; output ignores EOS, greedy.

Fixed 8192-token input SHA256: `defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`.

| Mode | Warmup prefill (s) | Formal prefill (s) (three times) | Median decode token/s | Client whole median (s) | Client first token median (s) |
|---|---:|---|---:|---:|---:|
| serial | 52.711 | 52.573 / 52.482 / 52.503 | 55.97 | 54.782 | 52.513 |
| batch-fp32 | 18.305 | 18.592 / 18.642 / 18.871 | 55.83 | 20.926 | 18.650 |

All 8 requests produced consistent output token IDs for all 128 tokens. Each new machine swap-out was 0; Metal allocation was 37, 995, 495, 424 bytes (35.39 GiB). Windows GPU sampled at 500ms; peak values are in the public JSON; sampling does not represent precise peaks for each allocation event. Process footprint was not collected in this group, so formal resource acceptance cannot be declared fully passed.

Serial tested first, batch tested later, no random order; both groups fully warmed up. One switch failed due to ordinary account WMI permission issues; attempts mislabeled as batch were aborted and excluded, with failure records retained. No fastest run was selected.

## Remaining bottlenecks

The median for the three formal stages is approximately:

| Stage | serial | batch-fp32 |
|---|---:|---:|
| Wait for frontend hidden | 31.77 s | 0.43 s |
| Mac backend 20 layer computation | 17.77 s | 15.01 s |
| Final state reception wait | 2.91 s | 3.22 s |
| State import | 0.019 s | 0.016 s |

Reception-side waiting is pipeline time, not equal to pure CUDA computation; summing the medians of each stage does not replace the whole median. The Mac computation difference of approximately 2.8 seconds between the two groups still needs further decomposition and cannot be entirely attributed to the CUDA kernel itself. After frontend acceleration, the Mac backend and the handover of the final state of 200, 704, 000 bytes have become the currently visible bottleneck.

This round's report covers the worker prefill stage and client TTFT; the coordinator t2−t1 required by the specification still needs complete instrumentation. The current approximately 439 token/s is on the 8K FP32 path, not the 128K target result, and cannot be directly compared as a same-configuration benchmark with historical Mac BF16 2363 token/s.

## Reproduction and Evidence

- `scripts/windows/benchmark-prefill-ops.ps1`: Independent primitive; does not load the complete model.
- `scripts/split/benchmark-prefix-chunk.py`: After stopping the verified CUDA agent, execute complete frontend numerical and time comparison; the same GPU cannot load a second model.
- Dual machines use the same release package, passing `--cuda-prefill serial` / `--cuda-prefill batch-fp32` respectively; after confirming the policy via actual ready logs, send `/v1/infer/stream`, `maxTokens=128, benchmarkIgnoreEos=true, profile=true` to fixed inputs. Each tier: warmup 1 + formal 3 runs, recording SSE time, token IDs, vm_stat, and GPU sampling.
- Public per-run summary: `site/cuda-chunk-summary.json`.
- Local raw evidence (unpublished): `artifacts/dual-host/chunk-e2e-serial-20261023T075202Z`, `chunk-e2e-batch-fp32-20261023T075943Z`, `cuda-prefill-chunk-387511b`; excluded attempts `chunk-e2e-batch-fp32-20261023T075701Z/INVALID.json`.
- 107 Mac unit tests, 31 pre-deployment integration checks passed; subsequently added 503/504 forwarding regression, 34 integration checks passed. Native frontend and independent operator checks as above. Both ends' AOT packages have been verified and deployed to cc19a65; subsequent HTTP error mapping fixes have not yet been deployed to this package.

Fixed targets remain: 131072 input prefill ≥1600 to 2000 token/s, 4096 output decode ≥37 token/s, total request ≤200 to 180 seconds. Next steps: optimize Mac backend block utilization, overlap state handover, and complete BF16 cross-host numerical verification; do not reduce quantization bit-width, do not relax numerical thresholds.
