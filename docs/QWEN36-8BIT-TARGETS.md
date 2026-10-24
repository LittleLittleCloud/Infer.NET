# Qwen3.6 35B A3B 8bit Performance Goals

Version v2, 2026-10-23 revision KV precision acceptance; v1's BF16 KV goals and unified numerical threshold for FP32 retained in Git history. Target devices M5 Pro 20 core GPU / 48 GiB and Windows RTX 4090 / 24 GiB.

The final target for this round is: **a complete 131072-token input, no prefix cache, and end-to-end prefill of 2000 token/s; the entire request with 131072 input tokens and 4096 output tokens must complete within 180 seconds.**
First pass the intermediate acceptance thresholds of 1600 token/s and 200 seconds for the long request, then continue optimizing toward the final target. Passing the intermediate thresholds does not establish final completion.
These are engineering targets, not promises or measured dual-host achievements. Dual-host execution and short/medium-input measurements already exist with 8bit weights and FP32 KV; Q8 KV and formal 128K acceptance remain incomplete.

## Fixed Model and Execution Plan

- Model: `mlx-community/Qwen3.6-35B-A3B-8bit`, revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine 8bit / group64. Retain the per-matrix quantization configuration already present in the checkpoint.
- Precision: Weights remain affine 8bit / group64; GEMM multiplication inputs are BF16, residuals, routing, Delta/convolution states, and accumulators remain FP32. **KV defaults to Q8, can explicitly select BF16 16-bit**; FP32 KV is only for comparison and current deployment rollback. The block format, quantization scale, rounding, and cross-machine byte layout of Q8 must be fixed before implementation; both ends must use the same format and reject precision mismatches; Q4/Q5 or lossy compression of other states cannot be used to replace this round's acceptance. Performance, memory, and quality results for Q8 and BF16 are reported separately, with no cross-precision speedup claims.
- Topology: 4090 stays resident for the first 20 layers and executes the front-end prefill; Mac stays resident with the complete model, executes the back-end 20 layers prefill, and after receiving the front-end KV and final Delta/convolution states, executes the complete decode. GPU does not dynamically swap expert weights in and out.
- Single request, concurrency 1, greedy, fixed output length. Throughput tests ignore EOS, quality tests handle EOS normally; fixed-length throughput does not represent natural response quality.
- The model is loaded and warmed up, but each formal request clears the KV/Delta state. The main metric is 0% prefix cache hit rate; speculative prefill, MTP, and answer caching are disabled.
- Use the existing home LAN, record wired/Wi-Fi, negotiated rate, and actual effective bandwidth. 10GbE is a separately measurable configuration, not a default premise; results for different links are saved separately.

The format to be connected for Q8 KV is fixed as: each K or V token's 512 elements are divided into 16 blocks of continuous 32 elements. Each block is arranged according to little-endian FP16 scale value, 32 signed int8 values, totaling 34 bytes; zero blocks are all zeros. The scale value is `round_fp16(max(abs(x))/127)`, non-zero quantized values are `clamp(round_even(x/float(scale)), -127, 127)`; non-finite inputs and invalid scale values are rejected. Each K or V token is 544 bytes, both together 1088 bytes, approximately 26.6% of the FP32 layout. The protocol can verify this layout, but CUDA/Metal native access is not yet connected; **current online deployment is still FP32 KV, do not treat this format description as an online feature**.

## Acceptance Metrics

128K in this document always equals **131072 tokens**, do not use 128000 as a substitute. The items in the table below must all pass simultaneously, you cannot just choose the best one.

|Metric|Stage Acceptance Line|Ultimate Goal|
|---|---:|---:|
|128K input + 128 output, end-to-end prefill|≥ 1600 tokens/s, i.e., ≤ 81.920 s|≥ 2000 tokens/s, i.e., ≤ 65.536 s|
|128K input + 4096 output, end-to-end prefill|≥ 1600 tokens/s, i.e., ≤ 81.920 s|≥ 2000 tokens/s, i.e., ≤ 65.536 s|
|Continuous decode of 4096 output after 128K input|≥ 37 tokens/s|≥ 37 tokens/s|
|128K input + 4096 output, client-side complete request|≤ 200 s|≤ 180 s|

Ultimate goal: 65.536 seconds prefill plus 4095 / 37 = 110.676 seconds decode, totaling about 176.212 seconds,
remaining about 3.788 seconds cover pre/post-processing and request transmission not yet included. The complete request is still based on client-side actual measurement, and cannot pass acceptance by adding two best-case scores.
The throughput of the last 1024 tokens of decoding must reach at least 90% of the throughput of the entire decode segment, to prevent the mean from masking the continuous slowdown in the latter half.

These thresholds are higher than current ideal split estimates, especially for the long output tier. Current 20/20 ideal estimate is about short output tier 1928 tokens/s, long output tier 1597 tokens/s,
pipeline overhead has not yet been included. Therefore, the ultimate goal requires further optimization of computation or workspace, and cannot rely solely on splitting the existing execution path across two devices.

## Single-Device Baseline and Regression Boundary

Historical results come from [8bit Verification Report](QWEN36-8BIT.md) and `artifacts/baselines/qwen36-8bit-summary.json`.
Short output tiers are one warm-up and three formal measurements; long output historical results have only one formal measurement.

|Input / Output|Historical Prefill|Historical Decode|Prefill Chunking|
|---|---:|---:|---:|
|8192 / 128|3.467 s, 2362.7 tokens/s|66.34 tokens/s|2048|
|32768 / 128|17.743 s, 1846.8 tokens/s|60.51 tokens/s|2048|
|131072 / 128|135.966 s, 964.0 tokens/s|37.52 tokens/s|2048|
|131072 / 4096|164.157 s, 798.5 tokens/s|38.95 tokens/s|512|

Long output historical worker request time is 269.355 seconds. Historical prefill is worker-internal timing, cannot be directly treated as this goal's distributed end-to-end timing;
before formally calculating the speedup ratio, you must re-test the Mac single-device control using the same timing method below.

Single-machine regression requirements: Under the same round and identical configuration, the increase in prefill latency across four tiers must not exceed 5%, and the decode throughput decrease must not exceed 5%.
If the new configuration reduces memory and increases overall throughput, changes in each component must still be reported; regression boundaries cannot be automatically exempted.
For dual-machine small inputs, explicit routing to a single machine is optional; the actual execution device must be recorded. Routing to a single machine does not count as dual-machine acceleration.

## Timing Definitions

Record the following timestamps for each request. End-to-end intervals use the monotonic clock of the same client or coordinator; subtracting uncalibrated timestamps from two different machines is prohibited.

- `t0`: The client starts sending the request. The input is ready; the time to read the test file is not included in the request.
- `t1`: The coordinator is ready with actual token IDs and starts dispatching to workers; model loading and warming up are completed before this point.
- `t2`: The Mac generates the first output token, and after completing the import of all states required for taking over full decode, the coordinator receives the ready confirmation.
- `t3`: The Mac generates the last output token.
- `t4`: The client receives the complete response.

**End-to-end prefill time = t2 − t1; throughput = actual input token count / this time.**
It includes dispatching, computation on both ends, hidden/KV/Delta transmission, CPU staging, GPU upload/download, state import, pipeline draining, and ready confirmation.
Timing must not be stopped prematurely before the 4090 front-end is completed, the Mac back-end is completed, or background state transmission has not finished.

**Decode throughput = (output count − 1) / generation interval between the first and last output tokens locally on the Mac.**
`t2` is used for coordinator-side prefill timing; the time for the first decode token is recorded separately on the Mac locally to avoid subtracting cross-machine clocks.
**Total request time = t4 − t0**, including templates, tokenization, RPC, and response transmission.
Report worker pure computation time and time for each stage separately, but these cannot replace the above primary metrics.
The current API is non-streaming; internal first-token readiness cannot be called user-visible TTFT. Once streaming is implemented in the future, the client-side first-token arrival time will be measured separately.

## Input and Re-measurement Rules

Use the current baseline input files, verify the total token count according to the actual template and tokenizer, including template tokens; requests with insufficient input, truncation, or length mismatch do not participate in acceptance.

|Input tokens|Path, relative to repository root|File SHA-256|
|---|---|---|
|8192|`artifacts/baselines/qwen36-context-fixed-data/8192-4096-prompt.txt`|`defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`|
|32768|`artifacts/baselines/qwen36-context-large-fixed-data/32768-8192-prompt.txt`|`4f2dfa83eb5f48eda2c8106a775d42d346d9dc72fa210c179c4db285b486f7f8`|
|131072|`artifacts/baselines/qwen36-context-plan-data/131072-8192-prompt.txt`|`0231f23f623809679e93ded59d9d70f04f7cf821ae5823c3996ae997b9cda767`|

Warm up completely once per tier and per configuration; perform formal measurements at least three times, including the 4096 output tier. Clear request state every time; warm-up and loading durations are reported separately.
Historical long-output results are measured only once, which does not comply with this specification. `benchmark-qwen36-8bit.py` now uses `--trials` simultaneously for the long-output tier (default three times); diagnostic runs with explicitly set fewer than three times cannot be marked as formal acceptance. Dual-machine end-to-end must still be accepted separately according to the timing and resource rules in this document; single-machine script results cannot replace them.

Performance thresholds are accepted based on the median of three measurements, while each prefill/total request time must not exceed 110% of the corresponding threshold, and each decode throughput must not be lower than 37 token/s.
Correctness, stability, and memory thresholds are accepted sequentially. Report the raw value, slowest value, and mean for each time; do not infer p95 from the three samples.
Tests are performed serially; the Mac is plugged in and set to High Power. Record temperatures/power modes on both ends, Windows display usage, background GPU workload, and link information.
In formal rounds, power supply changes, model configuration changes, or input errors invalidate the entire set; reasons must be retained; valid slow rounds cannot be deleted.

Record model revision, tokenizer/template/token IDs hashes, binary hashes on both ends, driver/OS versions, chunk length, precision, time for each stage, output token IDs, and memory/swap data.
Long outputs allow optimizing chunk length, but the old configuration comparison must be retained; the 128 output tier cannot replace the long-output tier.

## Correctness and Resource Thresholds

- **Same-precision correctness**: The FP32 diagnostic path uses independent FP32 references, adhering to relative L2 < 1e-4 and max absolute < 0.01. The Q8 formal path must verify quantization bytes and scales block-by-block against independently defined Q8 KV references, and validate CUDA frontend, layer boundaries/state, and full logits after cross-machine import under the same Q8 semantics; the BF16 optional path similarly uses BF16 KV references. Independent references and same-precision tolerances are fixed and published before formal acceptance; one cannot use FP32 tolerances to directly constrain Q8/BF16's normal quantization drift relative to FP32.
- **Cross-precision quality**: Layer/state/logits of Q8, BF16, and FP32 relative to L2, maximum absolute difference, first greedy divergence, and long-context task results are reported as-is, but a unified cross-precision L2 truncation value is not treated as proof of quality. Using pre-fixed short prompts, 128K retrieval/consistency prompts, and long-output tasks, comparisons are made against BF16/FP32; the task set, scoring methods, and acceptable degradation magnitude must be independently fixed before performance acceptance, without backfilling based on this round's results. Currently, there are no qualified Q8 quality results, so formal acceptance cannot be marked as passed.
- Existing 8 FP32 greedy reference tokens continue to be used for FP32 regression; Q8/BF16 and FP32 are not required to be token-by-token consistent; long requests of each precision must be deterministic upon repeated runs. If long greedy sequences of the same precision across different backends diverge, the first point of divergence must be located and state/logits verified; acceptance cannot be marked as passed until explained.
- Retain real RPC checks such as short-reference consistency after long requests, long-range checksums, and overflow rejection; in addition to fixed output stress tests, measure natural EOS to avoid proving only throughput.
- Formal rounds must have zero OOM, zero crashes, zero NaN/Inf, zero silent truncation, and zero CPU weight offload. CPU file reads during model loading do not constitute weight offload.
- 4090 full-device peak VRAM target ≤ 22 GiB, including system display usage, reserving at least 2 GiB for 24 GiB cards; device total peak and worker allocation must be reported.
- Mac Metal allocation peak ≤ 38.5 GiB, and process footprint and system memory pressure must be reported; total allocation cannot replace per-GPU submission residency safety checks.
- For each steady-state formal request, the whole machine's new swap-out ≤ 512 MiB; exceeding this means the group fails resource acceptance. This count cannot be attributed to a single process; background environment records must be retained. First-load/warmup swapping must be disclosed completely and separately; it cannot be marketed as "no swap".

## Scope and Completion Definition for This Round

This round's acceptance scope covers 0% cache hit, 128K full input, longest 4096 fixed output, 20/20 prefill sharding, and Mac full decode.
90% prefix hit, 256K input, 8192/32768 output, 24/16 sharding, and layered decode are not included in this round's performance commitments; baselines and targets will be established separately later.

Correctness and resource thresholds must be passed first before judging performance. Passing all stage thresholds only marks "Stage Acceptance Passed"; only when all final targets are passed can it be marked "This Round's Performance Target Completed".
This document only fixes acceptance specifications; it has not yet integrated an automatic judge, nor has it added new performance measurements.
Estimates, third-party results, lower precision, or cache-hit results cannot be used to update to "Passed".
If hardware measurements prove the target is unattainable, retain failure data and bottleneck analysis, explicitly revise the target version and reasons, and do not quietly lower the threshold.
