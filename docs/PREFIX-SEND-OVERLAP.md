# CUDA computation and hidden sending overlap

2026-10-23, following backend 128-token batching, added `cuda-send-overlap=on|off`. After A/B testing the same package, selected **on**: 8K prefill median 17.365 → 16.624 seconds, reduced by 4.27%; client entire 19.647 → 18.956 seconds, decode remains about 56 token/s. Model remains fixed affine8/group64, FP32 GEMM/KV; not BF16 or 128K acceptance.

## Implementation and Security

A production task exclusively occupies a CUDA session, computing and reading back 32-token blocks sequentially; a sending task transmits via the existing TLS/HMAC protocol in the original order. At most two frames are queued, plus the currently producing/sending frame; the entire request is not cached, and concurrent execution on the same GPU model is not allowed. Finally, 40 state tensors are handed over after all hidden states have been sent; no KV transfer occurs early in this round.

On cancellation, send error, or GPU error, queue cancellation is performed first and the production task is waited for before returning to the caller, avoiding a race between the next reset and in-flight GPU calls. There is no background retry. Four new tests cover ordering, backpressure, join/dispose after cancellation, and production exception propagation; 111 unit tests on this machine passed; 36 HTTP integration checks passed.

## In-package actual tests

Both ends Native AOT `2760e449364fbfd18f192d80aa66e41eb734a5da`; Mac process and configuration unchanged, only restart Windows agent to switch explicit toggle. Mac backend 128, scratch512; CUDA layer-wise batch-fp32, wire32. Fixed 8192 input, 128 output, greedy ignores EOS, reset each time, 0%prefix cache. Each tier warmed up 1 times, formal run 3 times. Mac AC / High Power, existing Wi-Fi/Ethernet LAN and TLS, no SSH tunnel used.

|Send overlap|Formal prefill seconds|Median prefill seconds|Client first token median seconds|Client whole run median seconds|Decode median token/s|
|---|---|---:|---:|---:|---:|
|off|17.592 / 17.365 / 17.228|17.365|17.374|19.647|55.88|
|on|16.624 / 16.547 / 16.790|16.624|16.636|18.956|55.84|

All 128 output tokens for 8 requests are identical to the previous serial baseline; each run adds a total machine swap-out of 0; each Metal allocation is 37, 572, 984, 832 bytes. Windows total GPU500ms sampling peak 19, 585MiB. Step-by-step warmup, resources, package hash, phase data see `site/send-overlap-summary.json`.

CUDA frontend compute/readback wall time still approx 13.4 seconds; original cumulative approx 0.8–1.1 seconds sending now overlaps with next block computation. `computeMs` is not pure GPU event time, `sendMs` includes framing/TLS, overlap time cannot be added as total duration. Frontend hidden all sent approx 14.35 →13.59 seconds; final state handover still approx 3 seconds. off first, on later, not random order. One driver startup attempt sent 0 inference requests when log file was not yet established, retained but not counted in data.

## Real cancellation and deployment acceptance

With overlap enabled, 8192-token prefill enters Busy then actively disconnects client after approx 1.07 seconds, fresh telemetry confirms both ends recovered after 2.38 seconds. Subsequently 8192 input /8 output same as baseline first 8 tokens, continuous Mac short requests also deterministic; Mac and Windows PIDs unchanged. Original evidence `artifacts/dual-host/overlap-cancel-20261023T084752Z`.

Additionally 15 deployment acceptance items all passed: real continuous requests, streaming consistency, cancel and resume after first token, natural EOS, overflow/auth and queue backpressure. Before switching also verified a8ce025 ReadyLocal behavior: Windows offline short requests 200 (0.293s), long requests 503 (5.005s). Windows still runs via foreground SSH control, does not impersonate SCM service.

CI three platforms passed: [2760e44 run](https://github.com/LittleLittleCloud/Infer.NET/actions/runs/39926055286). Virtual GPU residency test covering boundary follows backend batching report; physical M5 actually runs this test.

## Reproduction and boundaries

Same package keeps Mac `suffix-chunk-tokens=128`, Windows respectively `--cuda-send-overlap off/on`, verify actual switch via agent Ready log, then execute `/v1/infer/stream`, `maxTokens=128, profile=true, benchmarkIgnoreEos=true` on fixed 8192 input, each tier complete warmup 1+ formal 3. Input SHA256 `defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`, revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`.

Local original evidence not public: `artifacts/dual-host/overlap-e2e-off-20261023T084130Z`, `overlap-e2e-on-20261023T084415Z`, `overlap-release-2760e44`. Public JSON contains no credentials or private network addresses.

From initial same-precision serial 52.503 seconds to current 16.624 seconds, 8K prefill cumulative approx 3.16 times, but 128K prefill≥1600→2000 token/s, 4096 output decode≥37 token/s, whole run≤200→180 seconds still pending acceptance; do not treat short request decode results as long context pass. Next step will overlap KV handover with computation, and verify longer input/output.
