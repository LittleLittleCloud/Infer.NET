# DFlash2 versus official MTP: experiment plan

[Issue #80](https://github.com/LittleLittleCloud/Infer.NET/issues/80) records the user's direct implementation and comparative evaluation authorization on 2026-10-24 America/Los_Angeles. The implementation remains a candidate; no merge, production adoption, or baseline update is authorized.

## Question and fixed sources

Can native parallel DFlash2 block drafting improve Qwen3.6 decoding beyond the current official MTP implementation, including verification, feature capture, cache maintenance, selector, and rollback costs?

- Main baseline: `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3`.
- MTP reference: immutable `17965ce1f43e37398441c6f192c0a331efab228d`, [PR #78](https://github.com/LittleLittleCloud/Infer.NET/pull/78), width 1. Its previous pilot regressions and failures remain in [results PR #79](https://github.com/LittleLittleCloud/Infer.NET/pull/79). This experiment authorizes a fresh comparison; it does not change Issue #76's failed G5 decision.
- Target: `mlx-community/Qwen3.6-35B-A3B-8bit@e06a74e6236a60c8367e1a3214e83d8b61b637b0`. Existing affine8/group64 matrix formats and tokenizer are retained; KV and recurrent state remain FP32. All 11 pinned inference files were freshly hash-verified.
- Draft: `incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b6923ad6c14cb1bb41c37a9041446496ab6`; 96 BF16 tensors, 526,251,520 parameters, 1,052,513,456-byte safetensors. The complete local file matches pinned LFS SHA256 `d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d`. All source values were independently checked finite.
- Algorithm reference: `z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a`. The checkpoint has six 2048-hidden layers, sliding attention, two-tap grouped dynamic convolutions, and a trained rank-256/top-16 candidate selector. Target post-layer taps are zero-based layers 1, 6, 11, 16, 22, 27, 32, 37. A block of 8 includes one verified anchor and seven proposals.

The product runs its own C#/Metal implementation. BF16 draft matrices retain their trained values; norm scales are direct BF16-to-FP32 values, unlike Qwen3.6 MTP's raw-plus-one convention. Arithmetic, activations, and draft cache are FP32. The target embedding and output head are shared. MLX/PyTorch references do not become serving backends. The minimal exact verifier and state-transaction dependencies ported from PR #78 must be disclosed rather than treated as merged code.

## Evaluation and interpretation

The approved plan first checks independent draft equations, actual target feature taps, exact multi-row target logits/state for rows 1/2/4/8, every rejection offset, sliding-window eviction, growth, poisoned invisible slots, output caps, EOS, and cancellation/reset. A separate NumPy reference has generated deterministic backbone/selector golden data; those data are not evidence that native outputs already match.

After correctness/resource safety, test DFlash2 blocks 4 and 8 on pilot5. Freeze the higher aggregate worker decode rate, tie choosing 4. Safe diagnostic measurements continue even when performance adoption thresholds fail. This deliberately avoids stopping before the user's requested comparison merely because a pilot speed margin is disappointing. Demonstrated unsafe resources or incorrect target continuation still stop evaluation and retain evidence.

The formal comparison uses fresh serial main, DFlash2-off, MTP-width1, and selected-DFlash2 HumanEval20 runs: Pi 1.4.0, seed 0, context 32768, maximum 4096 output tokens/request, 300 seconds/task, zero prefix cache, full local M5 Pro 20-GPU/48-GiB Metal, AC/High Power. Identical fixed 128/4096-output controls use warm 1/formal 3; common allocation warmup equalizes initial target capacity. Captured identical-request replay is separate from independent agent sessions. No selective retries or exclusion of failed tasks.

Pilot order: 156, 94, 72, 80, 149. The 20-task order adds 2, 35, 32, 146, 137, 97, 154, 6, 161, 115, 131, 162, 50, 47, 148. HumanEval dataset revision `b9453e8f1865df0a0057a1383767caa400acfbcf`, gzip SHA256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`. Official assertions use `python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd` and an actual failing negative control.

Worker prefill is sum(input tokens)/sum(prefill seconds); decode is sum(N−1)/sum(actual first-to-last output generation seconds). Include the entire draft/selector/target-verify/cache/rollback pipeline. Prompt bootstrap belongs to prefill/TTFT. Client TTFT records first valid text/tool-call output, not a role event or internal worker readiness; the two loopback observer hops are included and Pi-to-observer dispatch is excluded. Report quality, metric coverage, complete request/token reconciliation, task wall time, accepted length, and component costs separately. Different agent trajectories limit causal attribution.

Expected gain is **to be measured**. Adoption criteria in Issue #80 are DFlash2 decode at least 1.20× fresh main and 1.10× fresh MTP, enabled prefill latency/TTFT regressions at most 5%, off decode loss at most 5% and off prefill/TTFT regressions at most 5%, zero additional off GPU allocations, complete correct 20/20 application results, exact controlled continuation/state, and full resource evidence. Threshold failures remain failures even when diagnostic runs continue.

Maintain Metal allocation at most 38.5 GiB, per-request whole-system new swap-out at most 512 MiB, zero OOM/crash/active NaN/Inf/silent truncation/CPU weight offload, and unchanged wired limit zero. Report process footprint, pressure, thermals, loading/warmup separately, and submission residency safety. Allocation-boundary/submission samples are not a continuous Metal-driver peak; missing peak evidence prevents full acceptance.

Splash's historical native-command speed, Q8 KV, quantized draft, and raised system memory budget are contextual references, not controlled single-variable evidence for this implementation. This comparison keeps draft BF16 and target FP32 KV to match the current MTP arm. Different proposal counts compare the available configurations, not an isolated algorithm effect at equal width. No Windows, sampling/tree decoding, precision changes, system-limit changes, generic formatter fixes, or 128K acceptance are included.

## Status and evidence

Implementation and independent review are in progress. Native correctness, serving performance, and formal HumanEval20 results are **not obtained**. No expected or synthetic performance points are plotted.

Local ignored evidence root: `artifacts/dual-host/issue80-dflash2/`. Pinned original algorithm sources, complete weight/source audits, independent NumPy golden fixtures, fresh original-service preflight, and immutable main/MTP package hashes are retained there. Raw prompts, generated solutions, credentials, and private configurations are not published. Every later measured phase requires its actual source/package hash and independent run ID.

The original service is currently unchanged. Future maintenance uses separate packages/ports and one model worker, verified process identities, and a restoration `finally` that checks the original Ready state, coordinator registration, two real chats, package/config hashes, and wired limit zero. Code and independent main-based results PRs require human review; this report does not grant merge or deployment permission.
