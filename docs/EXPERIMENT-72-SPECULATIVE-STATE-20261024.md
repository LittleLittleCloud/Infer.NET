# Experiment #72: speculative state transactions

Evaluation is in progress. [Experiment #72](https://github.com/LittleLittleCloud/Infer.NET/issues/72) was explicitly approved by the user on2026-10-24. [Code PR #74](https://github.com/LittleLittleCloud/Infer.NET/pull/74) implements the default-off synthetic proposal correctness reference; it is not a real DFlash2 draft and has no acceleration claim. This independent results branch starts from main and can be reviewed separately from implementation.

Baseline main:`4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3`. Candidate:`afdb36fd1106cf29b19b66ac94745dda481fc429`. Windows is not participating. Model remains fixed affine8 revision`e06a74e6236a60c8367e1a3214e83d8b61b637b0`, FP32 KV/state, fully local Metal, fused-fp32 attention, managed prefill chunk32 and scratch128. No system GPU-limit or precision change was made. The original worker will be restored after evaluation.

Implementation and corrective review used GitHub Copilot CLI1.0.103, usage-confirmed`claude-sonnet-5.5`. Independent CPU validation passed430 unit tests and70 HTTP/process integration checks. GPU/model fixture results, fixed5/20 cohorts, official assertions, package hashes and service restoration will be appended after they complete; no missing metric is filled with zero or an estimate.

## Retained setup failures

[Setup-failure summary](evaluations/issue72/setup-failures.json):

- An early parent scheduler treated the harness's incremental`summary.json` as proof the first Pi20 run had exited. Fixed-output fixture requests entered the same serialized worker queue. The extra fixture client was stopped; that cohort and interrupted fixtures are retained and invalid for formal A/B attribution. The cohort completed19/20 Pi tasks with one stream-finish failure on HumanEval/149. No official-assertion result is implied by Pi completion.
- The next attempt started before the observer coordinator registered the ready worker. Three initial requests returned503 without inference. The attempt was interrupted and retained. A replacement cohort starts only after authenticated worker and coordinator readiness plus a real streamed warmup are verified. Raw observations are selected by explicit run windows, preserving the excluded startup/warmup rows.

These are parent evaluation orchestration failures, not evidence of a candidate kernel regression. They will not be silently replaced by later successful runs. [Current evaluation status](evaluations/issue72/status.json).

## Measurement boundaries

For clean baseline/candidate-off, collect complete worker request timings and aggregateΣinputTokens/ΣprefillSeconds andΣ(N−1)/ΣdecodeSeconds. Report coverage, official assertions, first-valid-output clientTTFT, request/task counts, Pi wall time, failures and per-request resources. Enabled synthetic runs retain cycle/resource/emission-lifecycle diagnostics; their standard decode metric remains unavailable when generation-equivalent timestamps are absent.

Fixed token fixtures compare tokens, terminal and checkpoint state digests/logits, rejection offsets, EOS/output-cap boundaries, cancellation/reset and poisoned invisible KV suffixes. Diagnostic copy/replay times are not acceleration measurements. Acceptance remains exactly the preapproved Issue plan, including20/20 official assertions, disabled-mode regression/resource limits and exact deterministic continuation. Failure or partial completion will remain explicit. HumanEval does not substitute for the fixed128K goals.

No merge or adoption decision has been made. This record will be completed with actual results and restoration evidence before requesting review.
