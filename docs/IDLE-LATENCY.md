# Idle-then-first-request pause: residency and page-switch diagnosis

2026-10-22 (America/Los_Angeles). Related Issues #9 / #13. Conclusion: These pauses of tens to ninety seconds are not the steady-state latency expected from an already loaded, warmed-up inference service. Current evidence strongly points to Metal resource re-residency and memory pressure; slow prefill for thousands of tokens is also due to a deployment compute path that is not batched. No fix has been implemented, and residency strategy A/B testing is not yet complete.

## Reproduction in Same Process

Existing dual-machine backend source `506481445d3670d129ff3d101251f96357efb971`, Qwen3.6-35B-A3B affine8 / group64, revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, FP32 KV. Mac M5 Pro 48 GiB / macOS 26.6.2, RTX 4090, existing home LAN. Mac plugged in, High Power not re-verified; RTX 4090 read-only snapshot shows 19427 MiB usage, 0% GPU. No system settings modified, other apps closed, backend replaced, or second model loaded.

After the service becomes Ready, the coordinator sends two consecutive `/v1/infer/stream` requests. For each new HTTP connection, the fixed system is `You are a helpful assistant.`, user is `你好`, with maxTokens=32 and profile=true, resulting in natural EOS. The model PID remains unchanged; both requests use 24 input and 13 output, with all token IDs consistent. Since TUI was not used, it indicates that pauses occur even without restarting TUI.

|Metric|First after idle|Immediately following second|
|---|---:|---:|
|Client first visible text|3.709 s|1.327 s|
|Internal prefill (including handover)|3.348 s|1.314 s|
|First decode call, i.e., output token 2|16.393 s|0.410 s|
|Client total time|20.270 s|1.903 s|
|Mac cumulative GPU execution time|225.76 ms|225.50 ms|
|Whole-machine swap-in page count conversion|4.201 GiB|4.828 MiB|
|Whole-machine swap-out page count conversion|0.699 GiB|0|

The output token 1 is a complete `你好`, not an empty UTF-8 fragment; the client received the first character within 3.709 seconds, then waited approximately 16.4 seconds to receive the second token. The entire wait cannot be called prefill. Starting from the first token 3, the rate recovers to approximately 13.6–14.2 ms/token.

These are only two diagnostic runs; no commitment can be made regarding p95, formal throughput, or residency optimization benefits based on this. The first run included 1 seconds of sample sampling, introducing observation overhead. vm_stat provides whole-machine counts and cannot attribute all paging to this process; page count × 16384 is a counting conversion factor, not the actual SSD transfer bytes. Decompressed page counts convert to approximately 30.6 GiB, which does not imply an additional 30.6 GiB of disk data was read.

## Where the pause occurs

Stack sampling at approximately the 8 second of the first request shows:

- Inference thread: `SplitService.Handle → Qwen36Gpu.Forward → di_graph_forward → q_complete → waitUntilCompleted`.
- Metal submission thread: `_submitAvailableCommandBuffers → IOGPUCommandQueueSubmitCommandBuffers → iokit_user_client_trap`.
- Concurrent swap-in pages continued to grow; reaching approximately 4.20 GiB by request completion.
- The wall-clock time for this first-step decode was 16392.604 ms, with GPUStartTime to GPUEndTime being 17.769 ms.

Therefore, the main wait lies in the Metal/driver submission and completion path, appearing concurrently with significant paging; it is not that this GPU kernel truly computed for 16 seconds. It is also not waiting for Windows KV: the source code completes state import before entering this Forward. The process was not restarted, capacity remained 6144, and the sampled stack is not in model loading, tokenizer, network reception, or KV Grow.

Mac model Metal allocation is approximately 35.22 GiB; the device recommendedMaxWorkingSetSize is 37.44 GiB, with physical memory 48 GiB. Before testing, vmmap's IOAccelerator graphics showed approximately 30 GiB in the SWAPPED column, while the system's total swap usage was approximately 11.6 GiB: these are not the same metric, so one cannot claim the model has 30 GiB of SSD data. Current evidence of memory pressure, compression, and swapping is sufficient to list residency issues as the primary suspect, but other driver waits have not yet been ruled out using residency switch A/B testing.

The sharding design amplifies this transition: during prefill, the Mac only executes the last 20 layers; the first-step decode executes all 40 layers on the Mac, expanding the range of weights that need to be accessed. Current Metal buffers use shared storage but lack explicit residency set management. Shared indicates CPU/GPU shared memory, which does not guarantee continuous residency during idle periods.

## Comparison with mainstream frameworks and Apple documentation

- [Apple residency sets documentation](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets): Resources are made resident by default upon submission, which may delay GPU execution start; requestResidency can be used to prepare in advance, but contention from other applications may still delay processing.
- [llama.cpp PR #17766](https://github.com/ggml-org/llama.cpp/pull/17766): 2025-12-05 merged residency keep-alive to handle the delay of re-wiring after idle. This is an existing engineering issue, not a normal inference compute cost.
- [MLX Issue #4609](https://github.com/ml-explore/mlx/issues/4609): Reports that resources lose residency after idle, causing the next operation to be significantly slower. This report is a reproduction on other devices and its time or multiplier cannot be directly applied to this machine.
- [MLX residency implementation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/resident.cpp) already provides configurable residency refresh; [set_wired_limit documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html) explains that this budget is used to keep memory resident. Only the mechanism was studied; MLX was not integrated into the product backend.

## Why prefill itself is also slow

Deploy `SplitService.Profile` fixed chunk=32, FP32, matrix=0, Delta sequential. CUDA `di_cuda_prefix_run` inner loop explicitly iterates over tokens using GEMV, and performs synchronous D2H on the hidden state for each token. 32 is the block length for transfer/call, which does not imply that 32-token batch GEMM has been executed. It does not inherit all paths of the single-machine large-block optimization, so the user-observed rate of approximately 150–164 tokens/s cannot be treated as the 4090 capability upper bound.

Furthermore, the state of the previous 20 layers is handed over with each request. Even with only 24 input tokens, there are 15 layers of Delta/convolution fixed states, totaling 32, 931, 840 bytes, plus 491, 520 bytes of KV, approximately 31.875 MiB; additionally, there is approximately 0.188 MiB of hidden state. A warm request includes approximately 1.3 seconds of prefill containing these computations, CPU staging, Wi-Fi/TLS transmission and import, representing a real fixed overhead. Without phased measurement, one cannot attribute all 1.3 seconds to the network, nor can one treat the 12–26 seconds for thousands of tokens as inherent model latency.

## Fix Priority and Acceptance

1. implements budgeted residency set management for Metal weights and workspace, verifying whether periodic residency renewal is needed after idle; warmup and readiness must cover full decode, retaining OS/other application memory headroom. Perform local A/B testing first, without blindly increasing global memory limits.
2. Complete metrics for reset, front-end computation, waiting for hidden state, back-end computation, state reception/import, Metal commit/wait, and first-text segmentation.
3. Change CUDA prefill to batch GEMM; re-verify the Mac back-end batch path and large blocks. For short requests, explicitly evaluate single-machine routing to avoid paying the fixed state handover cost every time.

Acceptance suggestion: Under the same model, input, precision, power supply, and background environment, request after being idle for 0/2/10/60 seconds, and compare TTFT, token 1→2 interval, subsequent decode, swap increment, and token IDs. Compare multiple rounds before and after residency to judge the benefit; do not maintain busyness by continuously sending empty inference to mask the issue.

[Public diagnostic summary](../site/idle-latency-summary.json). The raw JSON, per-second vm_stat, sampled stacks, vmmap, and decode logs are in the local ignore directory `artifacts/dual-host/idle-latency-20261023T061238Z`, with no public process mappings or local path details. The formal performance threshold has not been reached, and no claim is made that the issue is fixed.

## Subsequent Corrections and Re-testing

See [Metal Residency A/B](METAL-RESIDENCY.md): The same-version switch comparison has been completed, and the first-step wait after short requests become idle is significantly reduced. The statement above regarding "Deploy Profile fixed matrix=0 / sequential Delta" was an inference based on managed assignments, later corrected by evidence that managed environment variables cannot update Unix native getenv; the native default is actually matrix=true / chunked Delta. The judgment regarding CUDA per-token GEMV and 32-token network calls still holds. Retain the original diagnosis, do not mix it with subsequent low-memory-pressure A/B tests.
