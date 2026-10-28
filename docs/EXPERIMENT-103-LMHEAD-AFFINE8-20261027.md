# Experiment 103: LM-head affine8 layout — pilot stopped on stream integrity

**Decoding improvement is Not Obtained.** The user-authorized, fully charged 35 W attempt passed native and full-model sequence/lifecycle gates. HumanEval5 ran on main and candidate off, but stopped on a malformed tool-call stream and token reconciliation failure in the off arm. Candidate-on performance, HumanEval20 and formal fixed-shape performance did not run. No measured sample was retried. Keep the candidate default off and unmerged.

Main's observed-request decode rate was **58.5122 tok/s**, with complete request-clock/token coverage, but Pi completed only 4/5 tasks. This diagnostic is not a complete successful five-task protocol result or an optimization comparison. Both arms' generated artifacts passed all five official HumanEval assertions.
- Experiment: [Issue #103](https://github.com/LittleLittleCloud/Infer.NET/issues/103).
- Candidate: [code PR #104](https://github.com/LittleLittleCloud/Infer.NET/pull/104), default off and unmerged.
- Independent [results PR #105](https://github.com/LittleLittleCloud/Infer.NET/pull/105).
- Measured candidate: `dee76d916361d4ad0d81cf44543cbee4b7a78488` (harness repairs; inference code unchanged).
- Main baseline: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`.
- Native correctness source: `a13abd9c5b3dc37a74e39cbbf82fcb3eda0f92ba`.
- Measured run: `battery-assisted-20261027-ready-fixed`, October 27, 2026, America/Los_Angeles. Initial native run: `gpu-correctness-20261027`.
- Current evidence: [sanitized trial summary](evaluations/issue103/battery-trial-summary.json), including per-task outcomes, request clocks, package hashes and restricted raw-evidence inventory.
- Initial evidence: [sanitized initial evidence](evaluations/issue103/initial-evidence.json). Raw logs are restricted local artifacts; the JSON records their SHA256 hashes and availability.

## Hypothesis and implementation

The affine8 classifier has 248,320 output rows, 2,048 input columns and quantization groups of 64. A tile-interleaved layout may improve coalescing and reuse input loads across four vocabulary rows. GitHub Copilot CLI 1.0.103, using the actual `gpt-6.1-sol` model through GitHub Copilot, authored the implementation and evaluation tooling. Codex researched, reviewed and coordinated the work.

The candidate retains the original code bytes, BF16 scale/bias bits and FP32 operation order. It does not requantize weights or factor the affine expression. The original classifier remains available for batched prefill. The single candidate layout, `tile4-block256-lane32-octet8-v1`, uses a separate 540,344,320-byte allocation (515.3125 MiB), with no repack scratch. The disabled path creates no extra packed allocation. `DEEPINFER_METAL_LMHEAD=tile4` is opt-in; unset or `off` selects the original path.

## Initial native correctness evidence

The initial run was a synthetic native operator correctness run on an Apple M5 Pro with 20 GPU cores and 48 GB unified memory. It did not load the pinned full checkpoint or measure performance. The 35 W adapter and 100% battery were recorded. The original Metal LaunchAgent was paused for exclusive GPU access, then its unchanged definition was restored.

| Check | Observed result |
| --- | --- |
| 63 small cases, original vs candidate logits | Bit-identical |
| Full vocabulary geometry, original vs independent reference | Relative L2 `8.19919096e-7`; max absolute `7.2479248e-5` |
| Full vocabulary geometry, candidate vs independent reference | Relative L2 `8.19919096e-7`; max absolute `7.2479248e-5` |
| Argmax ties, vocabulary boundaries, NaN and infinity controls | Passed |
| Reset/lifetime, selector and disabled-path allocation checks | Passed |
| Extra candidate payload | `540344320` bytes |
| Candidate scratch | `0` bytes |
| Full-checkpoint logits and internal states | Not Obtained |
| Full-checkpoint generated sequences | Not part of this initial run; subsequent gates passed below |

The full-vocabulary synthetic check is not evidence of full-model correctness. CPU byte-roundtrip tests, native/Metal compilation, managed tests and site checks also passed during implementation. An initial managed build without restored assets failed; a locked restore and subsequent build passed. Failed attempts remain in local evidence.

## Requested performance comparison

The registered plan compares fresh main, candidate off and candidate on under the same Qwen3.6-35B-A3B affine8 revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, FP32 KV and state, Pi 1.4.0, seed 0, context 32,768, output limit 4,096 and 300-second task timeout. The dataset SHA256 is `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`. Windows is outside this Mac-only experiment.

The source-pinned HumanEval driver, standard performance summarizer and Docker evaluator come from tools commit `349400b92b9de5efb88d73ce8cf5931f94f9e7b7`. The pilot has five fixed tasks and the formal cohort has twenty. A slow but otherwise valid pilot does not by itself cancel the requested formal cohort. Fixed-shape controls cover 21/128, 8192/128, 32768/128, 8192/4096 and 32768/4096, with excluded warmups and three formal repetitions.

Worker decode throughput is `sum(outputTokens - 1) / sum(decodeSeconds)`. Pi wall time, observer TTFT and worker prefill/decode are separate metrics. Missing clocks or token reconciliation failures invalidate the aggregate rather than becoming zeros. Correctness and resource gates precede a performance claim.

| Requested result | Main | Candidate off | Candidate on |
| --- | --- | --- | --- |
| HumanEval5 whole-protocol worker decode | Not Obtained | Not Obtained | Not run |
| HumanEval5 observed-request decode diagnostic | 58.5122 tok/s | Invalid token reconciliation | Not run |
| HumanEval5 Pi task completion | 4/5; one timeout | 4/5; one stream failure | Not run |
| HumanEval5 official assertion pass rate | 5/5 | 5/5 | Not run |
| HumanEval20 worker decode and official pass rate | Not Obtained | Not Obtained | Not Obtained |
| Fixed-shape prefill/decode | Not Obtained | Not Obtained | Not Obtained |
| Actual decoding change vs main | — | Not Obtained | Not Obtained |

The trend dataset retains both attempted arms and official assertion results, with throughput chart values null. Neither incomplete protocol becomes a successful speed point. The implementation has correctness evidence but no valid performance comparison.

## Initial restoration and preflight history

The original service remained build `0.3.0+cd6f7da401fadf044dac13e87b415d27bb5eadfb`. Bootstrap exited successfully, and its plist, worker configuration, executable and native-library hashes were unchanged. Health reported `state=Ready, ready=false` both before and after the check. No successful real-request restoration is claimed. The wired limit remained zero; the coordinator, Windows service and power settings were not changed.

The candidate Native AOT package at `f595ddf672579143683ea2e12e08c0fd586bed61` built successfully and passed all nine file-hash checks. All 32 CPU/synthetic harness tests passed. The read-only preflight verified both packages, model files, fixtures, Pi, dataset and source-pinned tools; its remaining blockers were the 140 W supply and fresh High Power evidence. Neither a runtime model request nor a service mutation occurred during preflight. Package hashes and versions are in the sanitized evidence.

At that stage, performance was blocked on power admission. The user subsequently authorized the exploratory battery-assisted policy below; it does not satisfy stable-140 W acceptance.


## User-authorized 35 W attempt: retained startup failure

After the initial report, the user explicitly authorized one attempt with the fully charged battery and existing adapter. Issue #103 plan revision 2 records a distinct exploratory `battery-assisted-35w` policy: initial SOC at least 99%, continuous 35 W AC with unchanged adapter, SOC at least 80% and no more than 20 percentage points of depletion, plus thermal, correctness and resource gates. The phase order prioritizes gates, HumanEval5, HumanEval20, then fixed controls. It is not stable-140 W acceptance.

Candidate `a3977c0842f4612488f15388f3215ae9b3ce9336` changes only harness policy and provenance relative to the prepared candidate. Copilot CLI/GPT-6.1 Sol authored it; 41 CPU tests and package verification passed. The inference kernel did not change.

Run `battery-assisted-20261027-attempt1` passed admission and native GPU correctness, then loaded main. Observer discovery failed with an `AttributeError`: importing the JSON `http` helper shadowed `http.client`. No external fixed/full-model/HumanEval request was issued, so there are no performance samples. The battery remained 100%.

Automatic cleanup rejected the macOS Python launcher/interpreter identity transition and did not claim restoration success. The owned processes subsequently exited. The parent verified all recorded children were gone, checked unchanged original files, bootstrapped the original LaunchAgent and observed `state=Ready, ready=false`, matching the original readiness state. No real-request restoration is claimed. The failed automatic recovery and subsequent parent recovery are both retained.

See [sanitized startup-failure evidence](evaluations/issue103/battery-startup-failure.json). Copilot repaired the observer and child-start identity handling with CPU-only loopback/subprocess tests. This initialization repair preceded measured samples; the failure produced no performance metric.


The next initialization, `battery-assisted-20261027-startup-fixed` at `ba8bf880d7b9ae83f1c4099c15c4bd8b0c37e3fa`, exposed a stale-registration race. Its first excluded warmup returned 503 in about 5.6 ms, with no token or worker timing. The observer checked a healthy lease without requiring the cloned registration to accept requests. No performance sample was produced. Automatic cleanup and unchanged-definition restoration then succeeded, returning to the original `Ready/ready=false` condition. [Registration-failure evidence](evaluations/issue103/registration-startup-failure.json) retains this separately.

Copilot tightened eligibility to the actual worker ID/PID/build, accepting status, Metal capability and chat model. The 49 CPU tests passed. Candidate `dee76d916361d4ad0d81cf44543cbee4b7a78488` has unchanged inference/native code and a verified package. Continuation `battery-assisted-20261027-ready-fixed` passed all main/off/on native and full-model sequence, natural EOS, cancellation and reset gates before entering HumanEval. The earlier startup failures remain retained; no measured samples were retried or discarded.


## Measured pilot outcome and stop

The final frozen candidate passed all 49 CPU harness tests, Native AOT packaging and all nine package hash checks. Its native library SHA256 is `2c8f58f5488301cfc66b4c66d0ed371f3c4e2a1427316938a2b84aabd10ee458`. Full-model gates passed for main, off and on: exact short 21/128 and 8192/128 generated sequences, natural EOS, cancellation and exact post-cancellation reset. Full-checkpoint logits/internal-state comparisons remain unavailable; these sequence gates do not replace them. The enabled selector logged the expected 540,344,320-byte extra payload; off logged zero.

The fixed pilot IDs were HumanEval/156, /94, /72, /80 and /149. Both arms attempted all five tasks and retained 21 observed requests out of 21 expected. Main's /149 hit the 300-second task timeout after producing an artifact that passed official assertions. Candidate off's /149 ended with an incomplete Pi run and a stream error, also after producing an artifact that passed. Thus official assertion success and agent task completion are different outcomes. All five artifacts in each arm were evaluated using the pinned official Docker evaluator; neither failed Pi task was excluded.

Main's standard request clocks and token counts reconciled, giving observed-request prefill **152.8622 tok/s** and decode **58.5122 tok/s**. The stricter whole-protocol aggregate remains null because a task was incomplete. Candidate off's rate remains null because token reconciliation failed. Pi task wall-time sums were 518.0150 s for main and 380.7163 s for off; these are incomplete-protocol observations and must not be compared as a speedup.

The final off request reported 3,645 input tokens, 151 output tokens and EOS at the worker. Pi recorded zero input/output tokens and `Stream ended without finish_reason`. The cloned coordinator logged `System.IO.InvalidDataException: Malformed model tool call.` in `ChatTools.ParseOutput -> CompatibleChat.Writer.End`; the stream ended without terminal usage/finish metadata. The controller stopped with `Pi/worker correctness mismatch; no whole-cohort claim`. Because the selector was off, this is not evidence of a tile4 numerical regression. Its underlying cause beyond the observed parsing/stream failure was not established.

Candidate-on HumanEval5, all HumanEval20 arms, formal fixed-shape repeats and the LM-head microbenchmark did not run. The user authorized one attempt; no measured cohort was automatically rerun, no failed request was discarded, and no kernel tuning occurred during the attempt.

## Power, resources and final restoration

The 35 W adapter remained unchanged; 136 sampled power records show battery charge moving from 100% to 91%. Fresh High Power UI evidence admitted the run. No thermal or resource stop was triggered. Request-sampled Metal allocation peaks were 37,286,100,992 bytes for main and 37,197,856,768 bytes for off. Maximum measured new swap-out within a steady request was 0 and 95,862,784 bytes respectively. These sampled allocator values are not continuous physical GPU-residency peaks. The off-arm swap observation is retained and must not be described as zero-swap execution.

Final cleanup restored the original service definition with unchanged plist/configuration/executable/native hashes and no restoration errors. Original build `0.3.0+cd6f7da401fadf044dac13e87b415d27bb5eadfb` returned to `state=Ready, ready=false`, matching its pre-existing readiness limitation. A successful real request was not verified. The original coordinator, Windows service, power settings and wired limit were unchanged.

## Research conclusion and next experiment

This attempt establishes implementation and sequence/lifecycle correctness evidence within the stated coverage. It does **not** establish a decoding improvement, regression, or adoption case. The extra 515.3125 MiB layout cost is known; its throughput benefit remains unmeasured.

Before another performance comparison, address malformed-tool-call error handling and terminal stream/usage semantics in a separate scoped change, preserving the failing evidence and source-pinned evaluator. Direct fixed-token main/off/on controls can then isolate LM-head performance from agent/tool-call failures. A fresh pre-registered run must still complete HumanEval5/20 and resource gates before an adoption decision; stable 140 W results remain a separate acceptance condition. No further GPU run or merge is part of this completed attempt.
