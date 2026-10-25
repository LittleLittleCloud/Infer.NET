# Experiment 33: native Q8 implementation fails independent full-model numerical gate

Date: 2026-10-25 America/Los_Angeles. [Issue #33](https://github.com/LittleLittleCloud/Infer.NET/issues/33),
[candidate code PR #85](https://github.com/LittleLittleCloud/Infer.NET/pull/85).
Baseline main: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`.
The human approved the concrete plan before implementation. The candidate is **not adopted**.

Native block32 Q8 storage, CUDA/Metal append/read paths, unchanged-byte incremental
transfer, explicit profile selection and reservation checks were implemented on
an independent branch. The default remains FP32. Codec correctness alone did
not establish full-model correctness: the independent same-Q8 model comparison
failed the pre-registered relative L2 <1e-4 / maximum absolute error <0.01 gate.

## Results

The full-model comparison used the pinned Qwen3.6-35B-A3B affine8/group64 checkpoint,
FP32 arithmetic/recurrence and the same block32 Q8 semantics on one M5 Pro 20-core
GPU /48 GiB. The independent reference used MLX FP32 model operations plus a
separate scalar CPU IEEE-half/round-even Q8 cache; the product remained native
C#/Metal. They ran sequentially. No model was replaced in the product backend.

The actual native full-model binary was built from `127d6a7`; the complete SHA
and executable/library hashes are in [evidence.json](evaluations/issue33/evidence.json).
Subsequent `cd16bc0` only adds complete trace selection to the reference script;
it is not presented as the tested product binary. Windows Q8 codec/build checks
were performed separately; no complete CUDA-to-Metal Q8 request passed here.

| Check | Observation |
| --- | --- |
| Independent fixture bytes versus Metal/CUDA codec | Passed, including zeros, signs, ties and invalid input/scale rejection |
| Metal Q8 attention versus double reference | Passed legacy, fused prefill and fused decode at lengths 5,2049,8193 |
| Metal imported-byte validation and growth preservation | Passed model-free checks |
| Managed regression | Mac132 passed; Windows131 passed,1 platform-specific skip |
| Independent full-model layer/logit gate | **Failed**; all121 comparisons retained in the public JSON |
| Worst sampled relative L2 | **0.002550493167395218**, threshold <0.0001 |
| Final logits maximum absolute error | **0.02856922149658203**, threshold <0.01 |
| Eight independent greedy IDs | Matched; this does not override numerical failure |
| HumanEval5/20, formal speed, TTFT, long-context quality/resource acceptance | **Not Obtained / not run**, numerical prerequisite failed |

The compared hidden states cover positions0,16,23 across40 layers, plus the final
full-vocabulary logits. The first failing *sampled* position16 layer15 has relative
L2 `0.00012403639366253007`; this does not claim to be the first divergent operation
over all positions. Full details: [independent comparison](evaluations/issue33/independent-comparison.json).

## State diagnosis

A later exclusive native prompt-state diagnostic exported the 17-token prompt's
frontend Q8 cache. Against independent reference bytes, layers3,7,11 had identical
K and V payloads. Layer15 K matched, while layer15 V differed by one byte: token1,
block8, byte13, independent code234 (signed -22) versus native233 (signed -23).
Layer19 already had additional scale/code differences.
[All byte counts and first offsets](evaluations/issue33/kv-byte-diagnosis.json) are retained.

This localizes a quantized-state difference. Small upstream FP32 differences
crossing a rounding boundary are a plausible explanation, not a proven root cause:
the pre-quantized native and reference operands were not captured at that exact
point. The standalone same-input codecs agreed. We do not relabel different-input
quantization as a codec pass for the full model or relax the approved threshold.

The complete-trace follow-up stalled after position1 and was interrupted. During
that follow-up a second diagnostic startup was attempted before the reference
ended; it exited without a captured status. Those attempts and logs remain local,
and no timing/resource attribution or acceptance claim uses them. A subsequent
exclusive state diagnostic produced the byte evidence above and was stopped
after its reference export, before any CUDA peer request. Its interrupted
listening lifecycle is not a passed distributed inference test.

## Decision and limits

Stop performance/adoption evaluation at the failed numerical gate. No HumanEval
or throughput point is fabricated, and no default or system-memory limit changes
were adopted. The resource thresholds and all12 pre-registered long-context
quality cases remain unpassed. The 128K target is unchanged.

Keep the code PR reviewable with its failure clearly visible. This independent
results PR preserves the implementation attempt and negative result; merging
records would not adopt the candidate. Further numerical redesign or a different
acceptance contract requires an explicitly scoped follow-up.

The original service restoration is recorded after the independent P0 experiments
finish; it remains pending while experiment84 uses isolated packages and ports.

Raw files are local-only under `artifacts/dual-host/issue33-*` in the experiment
checkout, including independent/native traces, prompt-state tensors, startup logs
and interrupted attempts. [Evidence hashes](evaluations/issue33/evidence.json)
identify usable artifacts without exposing private configuration or credentials.
