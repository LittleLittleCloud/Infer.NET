# Qwen3.6-35B-A3B: 8bit Verification and Dual-Machine Capacity Re-evaluation

2026-10-22, M5 Pro 20-core GPU / 48 GiB, macOS 26.6.2, AC / High Power.

See [8bit Performance Targets v1](QWEN36-8BIT-TARGETS.md) for the fixed performance thresholds and timing rules for subsequent dual-machine implementations. This document retains existing measurements and capacity estimates, which do not serve as evidence of dual-machine compliance.

## Checkpoints and Implementation

`mlx-community/Qwen3.6-35B-A3B-8bit`, fixed revision
`e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine 8bit / group64.
Downloaded files are verified one by one for LFS SHA-256 or Git blob hash; LICENSE and provenance are retained.
The checkpoint's MLX format is read by C#, but actual product execution remains Native AOT + custom Metal kernels.
MLX is used only for independent numerical reference, not for product requests or the performance measurements below.

This time, the 8bit path for embedding, classifier, batch GEMM, shared experts, and routing experts is completed,
while retaining 5bit regression. Model quantization configuration and single matrix coverage values will be verified.
KV capacity changes from 2 power-of-two growth to 2048-token pages, avoiding allocating 256K KV for just 128K input plus output.
The TUI text removes the hardcoded “5-bit”.

## Correctness

Independent FP32 reference vs. custom engine: 120 group layer-by-layer state, full logits, 8 greedy tokens.
8bit layer-by-layer max relative L2 = `3.559540353840437e-6`,
logits relative L2 = `2.2151446631205613e-6`; 8 output tokens are completely consistent.
Thresholds remain relative L2 < 1e-4, max absolute < 0.01.
5bit actual model regression also passes. Short numerical reference does not equal verification of long-context semantic quality.
This proves the implementation reproduces this 8bit checkpoint, not a task quality evaluation of 8bit against the original BF16 model.

Release build and Native AOT publish successful; 68 .NET unit tests passed.
Metal checks cover 5/8bit GEMV, FP32/BF16 expert GEMM, M5 packed/fused paths,
attention, DeltaNet, and KV page boundaries; raw logs for each check are in `artifacts/qwen36-8bit-*.log`.
8bit reference and comparison results are in `artifacts/reference/qwen36-8bit/`.

## Measurement Standards

Single worker, single request: load the model, warm up, then measure; clear KV/Delta state for each request,
**no cross-request prefix cache hit**. Weights 8bit, GEMM inputs and KV are BF16,
residuals, routing, Delta state, and accumulators remain FP32. Attention query batch=128,
key tile=128, partition=1024; hybrid dense GEMM, packed/adaptive/fused experts, device DeltaNet.

128 fixed outputs measure 127 decode intervals. Long outputs also ignore EOS to obtain fixed-length throughput,
which cannot be treated as the model naturally generating such a long meaningful answer. Prefill does not include model loading;
the GPU residency overhead for the first request is kept separately in the warm-up record.
128-output each tier involves one warm-up and three measurements. 4096-output uses 512 chunking instead,
with one complete warm-up followed by one measurement, listed separately for capacity/continuous decoding verification, not mixed with 2048 chunking calculations.

Results and original requests are in `artifacts/baselines/qwen36-8bit-*`.
System swap observations are whole-machine counts; you cannot attribute page-ins from other processes to the worker.

Final independent classifier submission version 128K / 128 output: one warm-up, three measurements,
prefill is **135.775 / 136.170 / 135.954 seconds**, mean **135.966 seconds**,
decode merged throughput **37.520 token/s**, three output token sequences are completely identical.
Metal allocation **37.938 GiB**, KV capacity **133120**.
First-round warm-up prefill **166.358 seconds**, cannot be mixed with steady-state calculations.
Three rounds of system new Swapouts pages are **0 / 0 / 8428** (16 KiB/page, last round approx 131.69 MiB);
Swapins are **29570 / 9315 / 12049**. These are results from the whole-machine environment, with user applications not closed.

Steady-state results under the same final binary and configuration (3 times per tier):

|Input|Fixed Output|Prefill (s)|Prefill token/s|Decode token/s|Metal allocation GiB|
|---|---:|---:|---:|---:|---:|
|8192|128|3.467|2362.7|66.34|35.351|
|32768|128|17.743|1846.8|60.51|35.868|
|131072|128|135.966|964.0|37.52|37.938|

Long output independent configuration (512 chunking, one complete warm-up, one formal measurement):

|Input|Fixed Output|Prefill (s)|Prefill token/s|Decode (s)|Decode token/s|Metal allocation GiB|
|---|---:|---:|---:|---:|---:|---:|
|131072|4096|164.157|798.5|105.123|38.95|37.541|

Worker first token to request generation end takes **269.355 seconds** (including pre-processing); warm-up prefill **206.333 seconds**.
KV capacity **135168**, KV bytes **2,768,240,640**. Two passes of 4096 output tokens are completely identical;
formal measurement 16 decode windows approx **38.30–39.19 token/s**.
Subsequently, 4 real RPC validations passed: 8K long-distance checksums, rejection of out-of-limit requests, consistency of short reference tokens after long requests,
determinism of repeated arithmetic requests. Validation does not equal 128K natural language quality evaluation.

Long output formal measurement whole-machine new swap-out **16639 pages = 259.98 MiB**, page-in **39669 pages**.
First complete warm-up adds new swap-out **620469 pages ≈ 9.47 GiB**, page-in **885357 pages**.
This indicates that loading/residing 8bit model in the normal application environment of this machine causes significant memory pressure, and it cannot be advertised as "no swap under 48 GiB".

8K last round whole-machine new 6340 swap-out pages (99.06 MiB), 32K three rounds are all zero.
All three output token sequences within each tier are consistent.
This round did not measure MLX's 8bit performance, so historical 5bit results cannot be used to judge the gap between frameworks.

## Memory boundaries of 128K

Calculated based on actual safetensors header, language weights **34.298735 GiB**,
all 10 layers full-attention 128K BF16 KV is **2.50 GiB**,
30 layers DeltaNet/convolutional state approx **62.81 MiB**, plus temporary buffers, runtime, and output capacity.
Metal returns recommended working-set budget of `40,200,896,512` bytes (37.44 GiB),
which is not 48 GiB exclusively available to the user.

The first round of 128K in the 2048-token prefill chunk reports Metal
`kIOGPUCommandBufferCallbackErrorOutOfMemory` at the last chunk; intermediate progress for 129024 tokens does not count as completion.
Original failure records are retained and their duration cannot be used as a baseline for success.
The 512-token chunk still fails at the final step. The ultimate fix was to submit the 8bit vocabulary projection separately:
Each request incurs only one GPU submission, inter-layer data does not return to the CPU, and computational precision remains unchanged.
The same submission no longer simultaneously references the prefill workspace and the full vocabulary matrix; 128K has completed
prefill and decode using 2048 chunks. Total Metal allocation can still exceed the recommended working-set value,
the key is the set of resources that must reside simultaneously at submission time; the recommended value is not an absolute upper limit for total allocation.
Failure records are retained separately in `qwen36-8bit-128k-128out` and `qwen36-8bit-128k-chunk512-pilot`.
The final binary measurements use the `qwen36-8bit-split-*` prefix, separated from old results.

If 128K + 4096 output still uses 2048 chunks, the reserved capacity increases from 133120 to 135168,
and the additional KV plus workspace will still trigger an OOM in the first submission. This failure record is in
`qwen36-8bit-split-128k-4096out`. Long output retesting simultaneously adopts **512 chunks + independent classifier submission**,
with the report prefix `qwen36-8bit-split-128k-chunk512-4096out`.
Therefore, "128K input is available" must be accompanied by output budget and workspace configuration details; it cannot be generalized to availability for arbitrary lengths.

## 4090 + Mac: Check capacity first

The goal is for 4090 to residently hold the first few layers, while Mac residently holds the full model and is responsible for the latter half of prefill and full decode.
Therefore, 24 GiB and 48 GiB are not treated as a transparent pool of 72 GiB VRAM.
After the GPU completes the first half of layer prefill, Mac must obtain the KV and final Delta/convolution states for these layers
to continue decoding the full model; transmitting only hidden vectors is insufficient.

For 128K input, the GPU front-end includes embedding; states are calculated via BF16 KV / FP32 recurrence:

|4090 / Mac prefill layers|4090 weights|128K state|Total|Remaining from 24 GiB|
|---|---:|---:|---:|---:|
|20 / 20|17.149 GiB|1.281 GiB|18.430 GiB|5.570 GiB|
|24 / 16|20.479 GiB|1.537 GiB|22.015 GiB|1.985 GiB|
|28 / 12|23.808 GiB|1.793 GiB|25.601 GiB|Exceeds|

The above margin has not yet deducted Windows display usage, CUDA workspace, transfer buffers, and allocator overhead.
Priority is given to verifying 20/20; 24/16 must be decided after measuring peak values; 28/12 no longer satisfies capacity.
Whether a single 8bit Mac can complete the target length is the prerequisite for this scheme.

The hidden vector per token boundary is 2048 × FP32 = 8192 bytes.
128K hidden totals 1 GiB; adding the GPU front-end KV and final recurrence states,
20 layers total **2.281 GiB**, and 24 layers total **2.537 GiB**.
Assuming an effective throughput of 110 MB/s or 1 GB/s, the link service times are respectively:

|Front-end|110 MB/s|1 GB/s|
|---|---:|---:|
|20 layers|22.26 s|2.45 s|
|24 layers|24.76 s|2.72 s|

This is byte count divided by assumed throughput, not a real-world home network measurement. Chunks can overlap transfer and computation, but one cannot simply add all network time
to the computation duration; nor can one ignore CPU staging, GPU readback/upload, and pipeline head/tail overhead.

If the 90% prefix already has correct caching at both ends, full capacity is still required, but only the new suffix hidden/KV
and final recurrence states need to be sent, reducing theoretical traffic to approximately 0.256 / 0.287 GiB (for 20/24 layers).
If the GPU lacks the corresponding prefix, states must first be filled; newly decoded states must also be synchronized in the next round.
The hit location must also have the corresponding DeltaNet/convolution state snapshot; one cannot arbitrarily fall back with only KV or the final recurrent state of a longer prefix.
Current products do not have this dual-end prefix cache, so this analysis only covers capacity and protocol, without claiming measured cache-hit throughput.

## How to determine if dual-machine implementation is worthwhile

Let the single-machine measured prefill be T. Assuming similar layer times can be split by layer count and both ends' pipelines are fully overlapped,
the Mac back-end for 20/20 is approximately T/2, and 24/16 is approximately 0.4T.
These are optimistic computational bottleneck estimates; one must still wait for 4090, network, and pipeline head/tail overhead, so they cannot be treated as dual-machine measurements.
The prefill acceleration upper limit for 20/20 is approximately 2 times, and for 24/16 approximately 2.5 times; actual results may be lower.

Under 128K, based on actual projection matrix shapes and causal full-attention QK/AV counts,
the full model requires approximately 2046.16 TFLOP; this excludes DeltaNet, dequantization, normalization, routing, and transmission overhead.
The first 20/24 layers require approximately 1023.08/1227.69 TFLOP respectively.
Using "this workload / measured Mac back-end estimated time" to derive the required effective throughput
is more useful than directly applying 4090's advertised peak. The true next hurdle is using the same 8bit format and chunks
to measure the actual duration, peak VRAM, and state export cost of the 4090 front-end.
The CUDA prototype should retain the same affine 8bit weights, BF16 multiplication inputs, and FP32 accumulation;
if changed to W8A8 or FP8 algorithms, re-verification is mandatory and cannot be directly treated as a throughput comparison at the same precision.

Substituting this round's **135.966 seconds**, 20/20 Mac back-end lower bound estimate **67.98 seconds**,
24/16 is **54.39 seconds**. To catch up with them, 4090 effective throughput for the above projection/attention work
must reach approximately **15.05 / 22.57 TFLOP/s**, and additionally bear other operator overhead not included.
These are the prototype acceptance goals, not existing actual measurement conclusions for RTX 4090.
The total service time of 22–25 seconds on a gigabit link can overlap with computation, so one cannot judge that gigabit necessarily yields no benefit based solely on this number;
10GbE provides greater margin, but whether it is worth purchasing still depends on actual CUDA front-end and network measurements.

The total request time for long outputs should be viewed in terms of `prefill + (outputTokens−1)/decodeThroughput`.
If decoding remains entirely on the Mac, prefill acceleration will not speed up this segment; when the 90% prefix hits,
the remaining prefill is shorter, and the total time savings in long-output scenarios will decrease further.
The time taken by the last 10% blocks of cold prefill serves only as an approximation of suffix computation volume, not as an actual cache-hit measurement.

Using the **actual long-output 512 block configuration** for a consistent estimation, prefill takes 164.157 seconds, decode takes 105.123 seconds:

|Scheme|Mac prefill estimated by layer count|Ideal total time maintaining existing decode|Relative to single-machine 269.28 seconds|
|---|---:|---:|---:|
|20/20|82.08 s|187.20 s|1.44×|
|24/16|65.66 s|170.79 s|1.58×, small VRAM margin|

This assumes the GPU and network can keep up and fully overlap, without yet accounting for pipeline start/end, state export/import, and scheduling overheads.
Therefore, prioritize running the 20/20 front-end CUDA micro-benchmarks; the above numbers are not actual dual-machine results.
The last 10% prefill of the 512 blocks took approximately **24.09 seconds**. Assuming correct future hits on the 90% prefix,
and continuing with 105.12 seconds of long decoding, the 20/20 ideal total time would drop only from approximately **129.21 seconds** to **117.17 seconds (1.10×)**.
This is a scenario derived from block durations, not an actual prefix-cache measurement.

The current CUDA back-end has not yet executed the front-end of this model; cross-machine layer splitting, KV/Delta state transmission,
and dual-end prefix caching have not been implemented. Therefore, current conclusions are used to filter prototypes and cannot claim that dual-machine acceleration has already been achieved.

If one still wishes to reduce the resident memory on the Mac, both ends must run their own layers during the decode phase,
rather than having the Mac retain the complete model to take over decoding. That is a different scheme, requiring measurement of per-token computation and network round-trips for both segments;
The 68/54 seconds estimate in this article applies only to prefill offloading and cannot prove the speed of layered decoding.

## Reproduction

First, run the downloader `scripts/download-qwen36-mlx.py --bits 8`.
Use an independent `artifacts/native/qwen36-8bit/deepinfer` to avoid replacing the existing 5bit service binary.
The original weights are located in `models/qwen3.6-35b-a3b-8bit-mlx`.

```sh
artifacts/baselines/mlx-env/bin/python scripts/reference-qwen36.py \
  --model models/qwen3.6-35b-a3b-8bit-mlx --output artifacts/reference/qwen36-8bit
artifacts/native/qwen36-8bit/deepinfer verify-qwen36 \
  --weights models/qwen3.6-35b-a3b-8bit-mlx \
  --reference artifacts/reference/qwen36-8bit/reference.json \
  --output artifacts/reference/qwen36-8bit/own --prefill
python3 scripts/compare-qwen36.py --directory artifacts/reference/qwen36-8bit
artifacts/baselines/mlx-env/bin/python scripts/benchmark-qwen36-8bit.py --label my-run
python3 scripts/summarize-qwen36-8bit.py
```

The benchmark requires Python tokenizers for input verification and does not import MLX for inference.
Existing results must be archived first; the drive refuses to overwrite reports with the same name. GPU workloads run serially.

The final 8bit artifact SHA-256:

- Native AOT host：`c6f28d325c85ebad4ba5f2767a9bc490d8b05a2d73ce5069dab6c07ebe8cb81b`
- Metal dylib：`4ea6561b9436f626c34c539c6680e03ae7f340423aafd6e3d20ecc8928d0794c`

All reports record hashes for the binary, tokenizer, input, and output. The summary script marks results corresponding to the current dylib,
preserving evidence of early successes and failures, and does not merge measurements from old binaries into new version scores.

After testing, restore the 5bit service on the 5100 port, preserving the TUI currently running for users;
8bit is saved in the aforementioned independent artifact. This time, the default model of the old launcher was not quietly changed to 8bit.

## Next Steps: Actual Link Transmission Baseline (2026-10-22)

[KV/Recursive State Sync Benchmark](KV-SYNC-BENCHMARK.md) has supplemented the transmission base previously untested in this article: Mac Wi-Fi ↔ Windows physical Gigabit Ethernet port, 128K to 20 layer complete state forward median 55.77 MB/s, 24.659 s, reverse 33.37 MB/s, 41.203 s. 40 composite state transmission verification passed; these results do not implement model state recovery or real prefix cache. Projecting based on the forward median rate, the hidden plus state link service takes approximately 43.91 s. The original 110 MB/s / 22.26 s in this article is retained as a hypothetical scenario and cannot represent the current link. Complete CUDA model frontend and dual-machine prefill remain untested.
