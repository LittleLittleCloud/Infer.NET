# Suffix receive overlap: no stable gain — 2026-10-23

An opt-in bounded reader let hidden/KV data arrive while Metal executed the previous suffix batch. It passed correctness and lifecycle tests, but **did not improve current end-to-end prefill consistently**. The integration branch removes the option and reader code while retaining this experiment and its evidence. The earlier hypothesis that the exposed receive wait could simply be removed was not supported.

Same source on both hosts: `638a45319a24f5f41a807b45c26f3c257e315813`. Affine8/group64, FP32 GEMM/KV, CUDA32/Metal128, incremental KV, CUDA send overlap on, fused FP32 decode, fixed inputs,128 output tokens, no cache, AC, one warmup plus three measured trials. All outputs agree with the retained baseline.

| Input | Off prefill s | On prefill s | Reduction |
|---|---:|---:|---:|
| 8192 | 14.306 | 14.282 | 0.169% |
| 32768 | 66.947 | 67.112 | -0.246% |

The first on8K comparison had a retained32K Windows allocation (20115MiB versus19585MiB in off8K). It is preserved with a comparison limitation, and is excluded from the reported paired result. Reloading the agent and restarting Metal produced the matched8K repeat above; both sides then had the same observed allocation/capacity. The original32K cohorts had matched allocations.

## What the wait actually represents

The Metal stage log showed roughly53.4s suffix work and13s exposed data waiting at32K. Windows stage logs then showed about65.6–65.9s in prefix computation plus readback, with send already overlapping it. At8K, prefix computation/readback was about13.5s, versus10.7s suffix work. Receive wait therefore includes waiting for upstream CUDA production; it is not all removable network latency. Windows computeMs is host wall time including D2H, not pure GPU time. No overlapping intervals are added as serial costs.

## Validation and implementation

The reader is single-owner, with two queued validated batches plus one being read and one being consumed (14MiB batch payload at128-token coalescing). The GPU remains single-owner. Recurrent-state handoff begins after the reader joins. Exceptions and cancellation join the reader before peer disposal. Native arithmetic, wirev4 and CUDA ABI6 are unchanged.

Mac123/123 tests passed; Windows122 passed and1 platform skip. Both Native AOT packages,39 HTTP integration checks and32 chat protocol checks passed, as did CI. All measured requests had zero swap-out; observed Windows peaks were19585MiB for matched8K and20115MiB for32K. Metal response samples remained under38.5GiB, not a continuous allocator peak claim.

Live prefill cancellation recovered in3.172s with the same model PIDs and matching next request. Fifteen deployment checks passed. Stopping the exact Windows agent during prefill returned502, Mac local inference stayed usable, a fresh agent reconnected, and the next128-token split output matched with the same Mac PID.

Public per-trial evidence: `site/receive-overlap-summary.json`; raw events, power, GPU samples, stage logs and input/result hashes remain local under the recorded run IDs. [PR22](https://github.com/LittleLittleCloud/Infer.NET/pull/22) preserves the original implementation and discussion. The formal Q8/128K performance gates remain open.
