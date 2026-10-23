# Incremental KV handoff — 2026-10-23

Same-package A/B reduced 8192-input / 128-output prefill from **16.839 s to 14.348 s** (14.79%). Client total fell from 19.130 s to 16.621 s; median decode was 55.83 versus 56.04 tokens/s. All 128 output IDs match across eight requests and the previous serial-prefix baseline. Relative to the earlier 52.503 s serial baseline, cumulative prefill improvement is approximately 3.66×.

These remain **affine8 / FP32 GEMM and KV diagnostic measurements**. BF16 / 131072-input acceptance is not passed, and no 128K result is extrapolated from this experiment.

## Change and mechanism

CUDA exports ten attention KV slices after each 32-token hidden chunk. One GPU producer assembles a chunk bundle; one sender transmits authenticated frames in order. The bounded queue holds at most two bundles, with one producer-current and one consumer-current bundle; cancellation joins the producer before any reset or disposal. Metal combines four wire chunks into a 128-token suffix operation, then imports their contiguous KV ranges. The thirty DeltaNet/convolution tensors transfer after prefill. Decode remains blocked until every required prefix tensor is complete.

All 200,704,000 state bytes still transfer. At 8192 tokens, 167,772,160 KV bytes overlap computation; 32,931,840 recurrent/history bytes remain at the tail. The measured final-state wait median falls from approximately 2.96 s to 0.42 s. This retains CUDA device-to-host readback, ordinary network buffers, and a CPU copy into Metal shared memory. It is lossless scheduling of existing bytes, not GPUDirect or compression.

## Same-binary measurements

| Mode | Warmup prefill s | Measured prefill s | Median client total s | Median decode tok/s |
|---|---:|---|---:|---:|
| final | 16.297 | 16.839 / 16.907 / 16.754 | 19.130 | 55.83 |
| incremental | 14.232 | 14.297 / 14.348 / 14.361 | 16.621 | 56.04 |

Source and both immutable packages: `788dd4a158cb6fee853f729b3f19032f3a774dbe`. Pinned model revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine8/group64, FP32 GEMM/KV, CUDA layer batch32, Metal suffix128, CUDA send overlap on, greedy, ignoreEOS=true, no cache hit. AC power, existing home-router LAN, exclusive GPU tests, one warmup and three measured trials per mode. Order was final then incremental, not randomized.

Peak Windows whole-device usage was 19,585 MiB; all eight requests had zero swap-out. Per-request Metal allocation, raw summary hashes, package hashes and client TTFT appear in `site/kv-streaming-summary.json`. Worker prefill includes computation and state handoff, but is not yet the formal coordinator t2−t1 measurement required by the fixed target document.

## Correctness and recovery

- Real CUDA weights, irregular 1/7/17/32-token chunks: FP32 and BF16 KV concatenated slices equal full export byte-for-byte; all forty state tensors and layer traces match the serial executor. Future positions and incorrect sizes are rejected. This does not establish full-model BF16 cross-backend accuracy.
- Independent whole-Metal reference, 84-token input / 8 output: final mode passed 1728 checks; incremental mode passed 1748 checks. Maximum relative L2 was 1.4992e-5 and maximum absolute error 0.0001234, within existing limits. All forty exported state tensors and all eight full logits are byte-identical between transfer modes.
- Native Metal tests exercise FP32/BF16 exact bytes, gaps, duplicates, wrong total, premature import, partial/full mixing, reset, and decode-before-complete rejection. Existing graph checks also pass.
- Mid-prefill client reset at 1.065 s: fresh Ready telemetry after 3.651 s, same Mac/Windows PIDs. Next 8192/8 request matches baseline and two local short requests are deterministic. An earlier recovery-check driver failed to parse a four-digit fractional telemetry timestamp on Python 3.9; that incomplete check is retained, the parser was fixed, and the check repeated in a new run.
- Mac 114 tests passed; Windows 113 passed, one platform skip; 37 HTTP integration checks passed; both Native AOT packages verified. GitHub build/test runs 39930041048 and 39930369775 passed.

## Operation and evidence

Experimental selector: `split-serve --kv-transfer final|incremental`; measured deployment explicitly selects incremental. This commit retains final as its CLI default. Both hosts require protocol v4 and CUDA prefix ABI 6; old peers fail authentication. Deploy or roll back the immutable pair together.

Native reproduction: `scripts/split/benchmark-prefix-chunk.py --weights ... --library ... --output NEW_DIRECTORY --kv-bytes 4` (repeat with 2). Run the corresponding AOT `split-infer --verify --kv fp32 --kv-transfer incremental` with the authenticated `split-worker` after stopping other full-model processes; repeat final with the same source and input. HTTP A/B uses the fixed 8192-token input SHA256 `defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d` and the parameters above.

Raw runs are retained locally under `artifacts/dual-host/kv-e2e-{final-20261023T091733Z,incremental-20261023T092055Z}`, `kv-check-788dd4a-{final,incremental}`, `kv-native-check-788dd4a`, and `kv-cancel-20261023T092341Z`. Native CUDA raw states are on Windows under `artifacts/split/kv-range-788dd4a-{2,4}`. Public aggregate evidence includes hashes, not downloadable raw tensors. [PR #20](https://github.com/LittleLittleCloud/Infer.NET/pull/20), [issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3).

Next: measure 32768 input / 4096 output, then address CUDA compute and BF16 numerical alignment. The current ~14 s CUDA prefix still limits 8K throughput; removing the handoff tail does not close the gap to the formal 128K target.
