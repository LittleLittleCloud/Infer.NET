# Experiment 103: LM-head affine8 layout — no gain in complete HumanEval5; HumanEval20 power stop

**The repaired trial completed all three HumanEval5 arms and showed no decoding improvement.** Main measured 60.2806 tok/s, candidate off 60.3149 tok/s, and candidate on 60.1165 tok/s: **on versus main -0.2722%**. All three completed Pi 5/5, passed official assertions 5/5, and had complete request-clock/token coverage. This single pilot does not establish a statistically meaningful regression or meet the predeclared +5% adoption threshold.

HumanEval20 started in the registered on/off/main order. The enabled arm completed 9/20 tasks before battery charge fell to 79%, triggering the unchanged 80% floor. The other two formal arms and fixed-shape performance controls did not run. **The formal20 decoding comparison is Not Obtained.** Keep the implementation default off and unmerged.

- [Experiment #103](https://github.com/LittleLittleCloud/Infer.NET/issues/103), [code PR #104](https://github.com/LittleLittleCloud/Infer.NET/pull/104), [independent results PR #105](https://github.com/LittleLittleCloud/Infer.NET/pull/105).
- New run: `battery-assisted-20261027-stream-fixed`, measured October 27, 2026, America/Los_Angeles.
- Main inference worker: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`.
- Measured candidate worker and common repaired API host: `dcf72d8769ed6836c3885f3f23d7b528324e2154`.
- [Sanitized request/task evidence and restricted raw hashes](evaluations/issue103/stream-repaired-trial-summary.json).
- [Plan revision 3 and user authorization](https://github.com/LittleLittleCloud/Infer.NET/issues/103#issuecomment-6461018668). The user separately confirmed continuing with 35 W.

## Repair and comparison boundary

GitHub Copilot CLI 1.0.103 using only GPT-6.1 Sol implemented the repair. An invalid generated tool call now emits genuine terminal usage before an explicit API error, instead of escaping as an uncaught exception and losing Pi accounting. Malformed calls are not executable, are not labeled successful, and the parser was not broadened. Missing or inconsistent worker accounting remains invalid. The exact historical malformed text was not retained; the sanitized regression reproduces its exception/accounting failure without claiming that text was recovered.

All three arms use the same repaired binary for both experimental loopback coordinators. The main inference worker remains the immutable main package. Thus main means **unchanged main inference under a common repaired API host**, not unchanged whole-stack main. Off/on isolates the layout switch; main/on additionally includes candidate worker plumbing. The deployed original coordinator was not changed. Separate coordinator and worker SHAs, package hashes and runtime identities are bound to the evidence.

The LM-head native code and all inference settings remain unchanged from the earlier candidate: affine8/group64, FP32 KV/operands/state, fused FP32 decode attention, managed prefill32/native scratch128, no prefix cache/speculation, the same model revision, dataset, Pi1.4.0, seed0, context32768, output4096 and task timeout300 s. Neither tuning nor measured retries occurred during the run.

Validation before measurement passed: 32 managed chat tests, 59 CPU harness tests, six termination tests including seven real Pi scenarios, 37 existing chat integration checks, solution build, Native AOT publish and all nine package hashes. The packaged host also passed all six termination tests. A parent invocation first used a nonexistent Pi path; it failed before its Pi scenario, was retained, and the corrected installed path passed. Native and full-model main/off/on short/8K sequence, EOS, cancellation and exact recovery gates then passed. Full-checkpoint logits/internal states remain unavailable.

## Complete HumanEval5 results

| Worker arm | Pi completion | Official assertions | Requests | Prefill tok/s | Decode tok/s | Decode vs main |
| --- | --- | --- | --- | ---: | ---: | ---: |
| main | 5/5 | 5/5 | 22/22 | 154.8814 | 60.2806 | baseline |
| off | 5/5 | 5/5 | 26/26 | 155.9005 | 60.3149 | +0.0569% |
| on | 5/5 | 5/5 | 16/16 | 157.6539 | 60.1165 | -0.2722% |

Both the standard source-pinned summarizer and strict controller reconciliation report complete metrics with no problems for all three pilot arms. No measured pilot request encountered a compatible API error. Rates use total worker input tokens/prefill time and total N-1 decode intervals/decode time; they are not means of request rates. Pi interactions and context lengths differ across arms, so application wall time is reported separately in the evidence and is not a decoding speedup. On versus off decode is also slightly lower; no reproducible gain is established.

The site retains valid pilot throughput points in a separate 35 W/common-host configuration. The x-axis identifies the measured inference worker; the shared API host identity is explicit in configuration and evidence. No candidate is promoted to adopted main.

## Interrupted HumanEval20 and power conditions

The registered sequence was gates, pilot main/off/on, formal20 on/off/main, then fixed controls. On completed HumanEval/156, /94, /72, /80, /149, /2, /35, /32 and /146. The next request was interrupted at the power stop and has no complete terminal worker accounting. All observed requests, including that failure, remain in raw evidence. Do not infer an independent stream-repair regression from this stop-associated incomplete request.

The supply remained 35 W with High Power verified. The controller sampled battery charge from 100% down to 79% and stopped at `2026-10-28T05:52:09.027814+00:00` with `Battery SOC must remain at least 80 percent`. This is an exploratory battery-assisted trial, not stable140 W acceptance. The pilot finished before the later power violation; the incomplete formal cohort has no aggregate speed claim.

After restoration, the pinned Docker evaluator checked existing artifacts without any new model request: all nine available solutions passed; eleven planned solutions were missing. Report this as nine verified artifacts from an interrupted twenty-task cohort, not a complete20 success rate or performance result. An initial parent evaluator invocation supplied a relative bind path, causing Docker exit125. That infrastructure failure is retained separately and is not a model assertion failure; using the required absolute path produced the nine genuine passes. The pre-arm positive/assertion-negative controls had already passed. No generated solution was changed or regenerated.

Maximum request-sampled Metal allocation was 37,286,100,992 bytes for main and off, and 37,738,201,088 bytes for on. Maximum observed steady-request new system swap-out was respectively 0, 180,076,544 and 18,677,760 bytes in the pilot. These are below the resource limits, but off/on are not zero-swap runs. Sampled allocation is not a continuous physical residency high-water mark. The enabled layout's separate payload remains 540,344,320 bytes (515.3125 MiB).

## Restoration and disposition

Automatic cleanup reported `childCleanup:GateError` and refused to reload the original service while ownership cleanup was uncertain. The exact child/probe failure was not recorded, so its root cause is not established. The parent subsequently verified all recorded children had exited, sent no signals, checked unchanged original file hashes, waited for resource quiescence and bootstrapped the original definition. Final hashes remained unchanged. Original build `0.3.0+cd6f7da401fadf044dac13e87b415d27bb5eadfb` returned to its prior `Ready/ready=false` state; successful real-request readiness is not claimed. Both failed automatic restoration and successful definition recovery are preserved.

Copilot subsequently added an owned-Popen exit recheck around failed/mismatched identity probes, preserving refusal to signal changed live identities and checks before escalation. The cleanup-only commit is `0bfc338e7e7a94a863cdb296b5232e4fc03dfbf6`; 70 CPU tests passed, including eleven new race/refusal/cancellation cases. All 269 snapshotted original trial artifacts remained hash-identical. The historical cleanup cause remains unproved, and this follow-up has no GPU measurement. [Copilot provenance](evaluations/issue103/stream-repair-provenance.json) records all three repair/tooling sessions.

The measured source remains `dcf72d8769ed6836c3885f3f23d7b528324e2154`. Any subsequent cleanup-only fix must be identified separately and cannot inherit this measurement identity. Original coordinator, Windows service, power settings and wired limit were unchanged.

The stream/accounting repair has CPU fault-path evidence and complete live pilot accounting. The affine8 layout has correctness evidence but **no measured pilot benefit**, while adding memory. Do not recommend adoption. The requested formal20 comparison remains unfinished because of the explicit power stop. Before another run, validate the cleanup follow-up under its appropriate acceptance scope and choose a pre-registered power schedule capable of completing comparable arms; do not relax thresholds or stitch these nine tasks into a complete cohort. No merge or GPU rerun is performed as part of documenting this result. Results validation passed 26 Python site tests, 27 JavaScript checks, the site build and whitespace/privacy checks. The JavaScript fixture loader initially exceeded its default 1 MiB buffer; Copilot set a bounded 8 MiB buffer without reducing evidence or assertions, and the checks passed.

---

The earlier incomplete attempt and its observations are preserved below. Statements in that section describe that earlier attempt, not the repaired trial above.

# Retained first attempt: pilot stopped on stream integrity

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
