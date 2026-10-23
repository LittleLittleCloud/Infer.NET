# Native Metal profile and bounded scratch — 2026-10-23

The service now reads its effective native Metal profile before becoming ready. The deployed profile is FP32 GEMM/KV with matrix batches, grouped experts, chunked DeltaNet and fused attention. A precision or scratch mismatch fails startup. This exposes a previously hidden discrepancy: managed environment assignments did not configure the native libc environment on macOS. The arithmetic actually used by the accepted service remains the same.

The opt-in `--metal-scratch-tokens 128` cuts temporary capacity from512 to128 tokens, matching the suffix chunk and short local prefill limit. It does not change the model, quantizer, arithmetic or Windows prefill. Source `73c3d5d`, pinned affine8/group64 model, fixed prompt hashes in [JSON](metal-scratch-summary.json). Same package, 512 then128, one warmup and three measured trials per shape, AC checked at trial boundaries; CUDA32/Metal128, incremental KV, fused FP32 decode, receive overlap off.

| Input/output | 512 prefill s | 128 prefill s | 512 client s | 128 client s | 512 decode t/s | 128 decode t/s |
|---|---:|---:|---:|---:|---:|---:|
| 8192/128 | 11.265 | 11.378 | 13.372 | 13.481 | 60.46 | 60.56 |
| 32768/128 | 53.072 | 52.991 | 55.952 | 55.807 | 44.78 | 45.43 |
| 32768/4096 | 53.457 | 53.062 | 143.683 | 142.951 | 45.39 | 45.59 |

At32K/4096, measured median response allocation fell **100.75 MiB** with the same KV capacity. The observed maximum response allocation fell from 36.920 to 36.627 GiB; these samples are not continuous peaks. Both modes stayed under38.5GiB in the samples. All output IDs match between modes for every trial, including every4096-token run. Last1024 client throughput was at least99% of full decode in the long runs. All long decode trials exceeded37 t/s.

Steady whole-system swap-out was 197.02MiB in one512 trial and110.92MiB in one128 trial; all other measured trials had zero. Both nonzero values remain below512MiB. These counters are system-wide; no model-specific swap saving is claimed. Windows whole-GPU sampled peak was20102MiB for matched32K/long cohorts. The8K cohorts differ in Windows capacity (19572 versus20102MiB), so their ~1% timing difference is not a clean scratch-only speed comparison. Mac 124 tests, Windows123+1 platform skip, both Native AOT packages,41 HTTP/32 chat checks and three-platform CI passed. Fifteen real deployment checks passed, including streaming, cancellation, natural EOS and bounded queue.

The128 configuration is running in the current deployment. The numerical and resource bounds verified here cover 8K/32K FP32 only. BF16 GEMM/KV and131072-token end-to-end gates in [the fixed targets](QWEN36-8BIT-TARGETS.md) remain unpassed. The separate partitioned-attention experiment passed FP32 but failed BF16; see [the investigation](CUDA-PREFIX-DIAGNOSTICS.md).
