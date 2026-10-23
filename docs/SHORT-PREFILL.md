# Short prefill routing and CUDA readback A/B

2026-10-23. Related #3 and #9. Measurement source: `a13e0a168def0e9e4d784ce930d80abe06630b78`; deployment candidate `2109a06` changes the unsuccessful batch-readback default back to token mode. Fixed affine8 model, FP32 KV, resident M5 Pro 48 GiB / RTX 4090, existing LAN, reset state on every request.

## Client-visible short request results

Same binary, only `short-prefill-tokens=0/128` changed. CUDA readback was explicitly `token` in both cohorts. Each of six cases had one warmup plus three measured requests per policy. All 48 generated token sequences agreed across repetitions and routes. These are natural-EOS outputs, not fixed-length throughput trials.

|Case|Input tokens|Forced split TTFT median|Local policy TTFT median|
|---|---:|---:|---:|
|hello|24|819.8 ms|91.4 ms|
|arithmetic|34|812.3 ms|148.1 ms|
|code|39|848.1 ms|156.8 ms|
|explanation|35|851.5 ms|150.6 ms|
|history|60|1066.4 ms|209.3 ms|
|long-control|351|2611.6 ms|2541.3 ms|

The 351-token control remained `cuda-metal-split` under both policies. The new policy avoids sending roughly 32 MiB of prefix recurrent/KV state for tiny prompts. It does not cache previous prompts or substitute another inference framework. RPC results and TUI metrics identify the route. Threshold includes the template and complete chat history.

A later split-to-local transition in the same process also returned the same 13 hello tokens. Its first request after idle/build activity took 0.596s total, the immediate repeat 0.271s; zero new whole-machine swap-in/out in those two additional diagnostics. Weight residency does not pin every scratch buffer: this is not a claim that all idle/memory-pressure latency has disappeared.

## CUDA candidate rejected as default

A 256 KiB device buffer collected final hidden vectors for one readback per 32-token chunk instead of one per token. Arithmetic and protocol layout were unchanged. With 131 synthetic token IDs and irregular chunk tails (1/7/17/32), hidden vectors and all 40 prefix state tensors matched byte-for-byte, for both FP32 and BF16 KV. Eight 512-token passes per KV (one warmup and three measured per mode, alternating order) had identical hidden hashes.

|KV|Token readback median|Batch readback median|Regression|
|---|---:|---:|---:|
|fp32|2.6022s|2.6947s|3.55%|
|bf16|2.6116s|2.7149s|3.95%|

Batch readback remains an opt-in diagnostic. Production keeps `cuda-readback=token`; fewer synchronizations alone did not improve this WDDM/GEMV path. The result is not evidence against batching GEMM: CUDA still computes tokens serially with GEMV. BF16 state equality here compares two readback modes of the same CUDA computation; it does not pass the outstanding CUDA-versus-Metal BF16 accuracy gate.

## Validation and remaining work

- 107 Mac tests, 30 CLI/RPC integration checks, 65 controlled Chat Completions / Messages / Responses SDK and TUI checks passed. Windows build ran 107 tests (Mac-specific tests return on Windows), CUDA operator checks and Native AOT packaging.
- The tested native executor and complete host use their own CUDA / Metal kernels. No precision downgrade or external inference backend.
- Raw requests/results and SHA256 inventory: [short-prefill-summary.json](../site/short-prefill-summary.json). Full logs remain local under the named artifact directories.
- Reproduce the route test with `scripts/benchmark-short-route.py --help`, running the two explicit policies with one resident model per device. Native readback driver: `scripts/split/benchmark-readback.py --help`; stop the CUDA agent before loading the standalone test model.
- Fixed long-input gates are unchanged: 131072-token prefill >=1600 then2000 tok/s; 4096-token decode >=37 tok/s; complete request <=200 then180s. None are passed by these short-request results. Next work is actual CUDA batched GEMM/Delta execution and consistent BF16 precision.
