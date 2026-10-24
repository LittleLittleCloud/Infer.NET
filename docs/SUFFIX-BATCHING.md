# Mac back-segment batching: 128 outperforms 32, 512 does not further improve the entire request

2026-10-23, following CUDA layer-by-layer batch execution, maintains front-segment 32-token wire chunks, where Mac merges multiple hidden blocks before executing the 20 layer. `suffix-chunk-tokens=32|128|512`, this round selects **128**. It does not change the model affine8/group64, nor does it change the deployment diagnostic precision for FP32 GEMM/KV.

Native scratch is set to 512 tokens via an explicit C ABI, avoiding behavior dependent on inconsistencies between .NET environment variables and the libc environment on Unix. Effective Metal allocations decreased from 37, 995, 495, 424 bytes in the previous version to 37, 572, 984, 832 bytes, a reduction of approximately 403 MiB. All three tiers in the formal A/B test use the same 512 scratch, and scratch changes are not counted as part of the batched gains.

## Values and Resources

Diagnosis first generated reference on full Mac model by 32-token blocks, then took the 20 layer hidden input to the backend, compared all 20 layer traces, 8 step complete logits and greedy output, and imported same frontend state to verify subsequent decoding. 234, 714 two actual templates covered small tail blocks, 128 cross-block, 512 cross-block; 32 comparison completely consistent, 128/512 max relative L2 was `1.1783111e-5`, max absolute `3.4570694e-5`, all passed fixed `1e-4 / .01` threshold, 8 greedy tokens consistent. No text similarity used instead of numerical verification.

All 128 output tokens from 16 valid requests of the complete 8K API are identical to the previous CUDA sequential baseline. Each new machine swap-out is 0, with Metal allocation at approximately 34.99 GiB, and the Windows full GPU 500ms sampling peak below 19, 585 MiB. Mac `vmmap -summary` is collected at the end of each tier, with physical footprint/peak recorded in JSON; the initial two tiers have peaks of approximately 35.6G. Sampling does not equal the precise peak of all allocation events.

## Same-Package Dual-Machine A/B

Both ends use Native AOT source `a8ce0250b959b9279be8ccce99e9922c07f47f22`, with the same CUDA agent; the Mac restarts each tier according to the new configuration. The model is warmed up once and run formally three times, resetting each time, with greedy fixed at 128 output and 0% prefix cache. Inputs remain fixed at 8192 tokens, SHA256 `defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`. Mac AC / High Power, with existing Wi-Fi ↔ Ethernet LAN / TLS, without using SSH tunneling.

|Mac Backend Block|Formal Prefill Seconds (Three Times)|Median Prefill Seconds|Client Median First Token Seconds|Client Median Total Seconds|Decode Median Tokens/s|
|---|---|---:|---:|---:|---:|
|32|18.125 / 18.402 / 18.408|18.402|18.414|20.682|55.94|
|128|17.314 / 17.134 / 17.168|17.168|17.181|19.457|55.78|
|512|17.536 / 17.978 / 17.621|17.621|17.632|19.904|55.86|
|128 Confirmation|17.355 / 17.142 / 17.403|17.355|17.366|19.628|55.82|

The initial round of 128 reduced 6.71% prefill time, and the confirmation round still reduced it by approximately 5.69%. The complete evolution is earlier sequential 52.503s → CUDA batch 18.642s → current approximately 17.2–17.4s; values across versions come with their respective configurations and cannot replace the same-package A/B in the table above.

The Mac computation for 512 itself is faster (median 8.12s, 128 is 10.84s, 32 is 14.98s), but the receiver waits approximately 6.75 / 3.51 / 0.47s respectively. Larger blocks increase pipeline fill and wait times, so one cannot simply choose the tier with the shortest GPU computation time. The final state reception still takes approximately 2.8s.

The order is first 32, then 128, then 512, and finally 128 confirmation, not random. A set of confirmation rounds potentially overlaps with physical GPU unit tests; the entire set has been excluded and separate exclusive confirmations have been made; excluded run `suffix-e2e-128-20261023T082656Z` and reasons are retained, without selecting the fastest single data point.

## Build, CI, and Reproduction

- The AOT packages at both ends are hashed in `site/suffix-batching-summary.json`; local unit tests passed on Mac and Windows with 107 each, and the 35 item's HTTP integration check passed. Two explicit scratch configurations were added to the native graph check, and the original 60 item's hardware check passed.
- CI `f751653` passed on three platforms: [run record](https://github.com/LittleLittleCloud/Infer.NET/actions/runs/39923894758). The macOS runner's actual name is Apple Paravirtual device, which cannot create a residency set; this hardware test case is explicitly skipped, and CI shows 106 passed / 1 skipped. The physical M5 machine still runs the same test case, with 107 passed / 0 skipped. Not all residency exceptions were captured and silently treated as passed.
- Single-machine diagnostics: After stopping the verified resident Mac model, run `deepinfer split-check-suffix --weights MODEL_DIR --prompt-file PUBLIC_PROMPT --output NEW_DIRECTORY` with process environment `DEEPINFER_METAL_KV=fp32 DEEPINFER_METAL_PREFILL_CHUNK=32 DEEPINFER_METAL_PREFILL_GEMM=fp32`; accept up to 1024 inputs, saving the original trace, state, logits, and comparison results. Do not load a second model on the same GPU.
- Full measurement: Configure `suffix-chunk-tokens` as 32/128/512 for the same package, call `/v1/infer/stream`, `maxTokens=128, profile=true, benchmarkIgnoreEos=true` with fixed inputs; warmup 1 + formal 3 per tier, verifying the suffixChunk in the actual PREFILL logs.
- Local raw evidence not published: `artifacts/dual-host/suffix-check-{9d3304b,a8ce025}`, `suffix-e2e-32-20261023T081727Z`, `suffix-e2e-128-20261023T082142Z`, `suffix-e2e-512-20261023T082422Z`, `suffix-e2e-128-20261023T082851Z`.

This round still outputs FP32 / 8K / 128. **BF16 cross-host values, 128K prefill ≥1600→2000 token/s, 4096 output decode ≥37 token/s, full process ≤200→180 seconds have not been accepted.** Do not run known 128K FP32 requests that exceed memory budgets, and do not relax precision thresholds. Next, overlap frontend computation and network sending, then handle the final centralized KV handover.
