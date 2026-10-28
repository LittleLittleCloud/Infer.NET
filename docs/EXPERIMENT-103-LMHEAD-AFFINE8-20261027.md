# Experiment 103: LM-head affine8 layout — initial correctness evidence

The candidate passed the native operator checks described below. **Decoding improvement is Not Obtained.** HumanEval5, HumanEval20 and fixed-shape end-to-end performance have not run. The detected 35 W supply does not satisfy the registered 140 W performance condition; the parent requested the physical supply change and continued implementation and CPU checks. No speed or adoption conclusion follows from this report.

- Experiment: [Issue #103](https://github.com/LittleLittleCloud/Infer.NET/issues/103).
- Candidate: [code PR #104](https://github.com/LittleLittleCloud/Infer.NET/pull/104), default off and unmerged.
- Main baseline: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`.
- Native correctness source: `a13abd9c5b3dc37a74e39cbbf82fcb3eda0f92ba`.
- Run: `gpu-correctness-20261027`, measured October 27, 2026, America/Los_Angeles.
- Evidence: [sanitized initial evidence](evaluations/issue103/initial-evidence.json). Raw logs are restricted local artifacts; the JSON records their SHA256 hashes and availability.

## Hypothesis and implementation

The affine8 classifier has 248,320 output rows, 2,048 input columns and quantization groups of 64. A tile-interleaved layout may improve coalescing and reuse input loads across four vocabulary rows. GitHub Copilot CLI 1.0.103, using the actual `gpt-6.1-sol` model through GitHub Copilot, authored the implementation and evaluation tooling. Codex researched, reviewed and coordinated the work.

The candidate retains the original code bytes, BF16 scale/bias bits and FP32 operation order. It does not requantize weights or factor the affine expression. The original classifier remains available for batched prefill. The single candidate layout, `tile4-block256-lane32-octet8-v1`, uses a separate 540,344,320-byte allocation (515.3125 MiB), with no repack scratch. The disabled path creates no extra packed allocation. `DEEPINFER_METAL_LMHEAD=tile4` is opt-in; unset or `off` selects the original path.

## What was measured

This was a synthetic native operator correctness run on an Apple M5 Pro with 20 GPU cores and 48 GB unified memory. It did not load the pinned full checkpoint or measure performance. The 35 W adapter and 100% battery were recorded. The original Metal LaunchAgent was paused for exclusive GPU access, then its unchanged definition was restored.

| Check | Observed result |
| --- | --- |
| 63 small cases, original vs candidate logits | Bit-identical |
| Full vocabulary geometry, original vs independent reference | Relative L2 `8.19919096e-7`; max absolute `7.2479248e-5` |
| Full vocabulary geometry, candidate vs independent reference | Relative L2 `8.19919096e-7`; max absolute `7.2479248e-5` |
| Argmax ties, vocabulary boundaries, NaN and infinity controls | Passed |
| Reset/lifetime, selector and disabled-path allocation checks | Passed |
| Extra candidate payload | `540344320` bytes |
| Candidate scratch | `0` bytes |
| Full-checkpoint logits, internal states and generated sequences | Not Obtained |

The full-vocabulary synthetic check is not evidence of full-model correctness. CPU byte-roundtrip tests, native/Metal compilation, managed tests and site checks also passed during implementation. An initial managed build without restored assets failed; a locked restore and subsequent build passed. Failed attempts remain in local evidence.

## Requested performance comparison

The registered plan compares fresh main, candidate off and candidate on under the same Qwen3.6-35B-A3B affine8 revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, FP32 KV and state, Pi 1.4.0, seed 0, context 32,768, output limit 4,096 and 300-second task timeout. The dataset SHA256 is `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`. Windows is outside this Mac-only experiment.

The source-pinned HumanEval driver, standard performance summarizer and Docker evaluator come from tools commit `349400b92b9de5efb88d73ce8cf5931f94f9e7b7`. The pilot has five fixed tasks and the formal cohort has twenty. A slow but otherwise valid pilot does not by itself cancel the requested formal cohort. Fixed-shape controls cover 21/128, 8192/128, 32768/128, 8192/4096 and 32768/4096, with excluded warmups and three formal repetitions.

Worker decode throughput is `sum(outputTokens - 1) / sum(decodeSeconds)`. Pi wall time, observer TTFT and worker prefill/decode are separate metrics. Missing clocks or token reconciliation failures invalidate the aggregate rather than becoming zeros. Correctness and resource gates precede a performance claim.

| Requested result | Main | Candidate off | Candidate on |
| --- | --- | --- | --- |
| HumanEval5 worker decode and official pass rate | Not Obtained | Not Obtained | Not Obtained |
| HumanEval20 worker decode and official pass rate | Not Obtained | Not Obtained | Not Obtained |
| Fixed-shape prefill/decode | Not Obtained | Not Obtained | Not Obtained |
| Actual decoding change vs main | — | Not Obtained | Not Obtained |

No throughput trend point is created for these missing measurements. The implementation commit and this record commit must not be mistaken for end-to-end tested code.

## Restoration and remaining work

The original service remained build `0.3.0+cd6f7da401fadf044dac13e87b415d27bb5eadfb`. Bootstrap exited successfully, and its plist, worker configuration, executable and native-library hashes were unchanged. Health reported `state=Ready, ready=false` both before and after the check. No successful real-request restoration is claimed. The wired limit remained zero; the coordinator, Windows service and power settings were not changed.

Next: verify the 140 W supply and fresh High Power evidence, freeze and verify the candidate package, execute full-model sequence/lifecycle gates, then the fixed controls and HumanEval5/20. Preserve this initial record when appending results. Until valid end-to-end data exist, keep the code unmerged and make no adoption recommendation.
