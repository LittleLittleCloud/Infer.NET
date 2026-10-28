# Issue #103: default-off LM-head affine8 implementation

Status: implementation candidate, **GPU correctness and performance untested**.
Author: GitHub Copilot CLI 1.0.103, GPT-6.1 Sol (`gpt-6.1-sol`), without
subagents. Baseline: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`.
Branch: `codex/experiment-103-lmhead-affine8`. No commit, push, PR, service
operation, full-model execution, GPU dispatch, or system-setting change is
part of this implementation phase. Runtime Metal library/pipeline compilation
does not submit command buffers.

The approved revision-1 plan and explicit user authorization are retained
locally at `artifacts/dual-host/issue103-lmhead-affine8/control/approved-plan.md`.
The experiment is [#103](https://github.com/LittleLittleCloud/Infer.NET/issues/103);
adoption still requires the approved independent correctness, resource and
HumanEval5/20 evaluation against fresh main. No measured speed or trend point
is supplied here.

## One layout/kernel variant

`tile4-block256-lane32-octet8-v1` is the only implemented variant. Production
selection requires exactly experts=1, N=248320, K=2048, bits=8, group=64.
Packing is a byte permutation, not requantization. For vocabulary row `r`
and column `c`, the code-byte offset is:

```text
((((r / 4) * 8 + c / 256) * 32 + (c % 256) / 8) * 4 + r % 4) * 8 + c % 8
```

Scale/bias element offset for group `g=c/64` is
`((r/4)*32 + g)*4 + r%4`, with two bytes per BF16 element. The single buffer
contains codes followed by scales followed by biases:

| Region | Byte offset | Byte length |
|---|---:|---:|
| Codes | 0 | 508559360 |
| BF16 scales | 508559360 | 15892480 |
| BF16 biases | 524451840 | 15892480 |
| Total | 0 | **540344320** |

N has no production padding. The shared pack helper supports padded
four-row tails only for independent operator tests; production rejects
different geometry. No packed storage is attached to other matrices.

Each SIMD32 processes four rows, keeping four FP32 sums per lane. Four
SIMD groups per 128-thread threadgroup process 16 rows. Lane `l` consumes
columns `block*256+l*8+j`, for blocks 0..7 and j=0..7, exactly the default
original vector-GEMV mapping and per-lane accumulation order. Its four
row octets are adjacent in storage; each FP32 input load is reused across
four rows in registers. There is no threadgroup scratch or extra reduction.
Each row retains a separate SIMD32 `simd_sum`. This is not the rejected #92
smaller-lane row-major approach.

The expression remains `(float(code)*scale+bias)*input`, accumulated in
FP32. Scale/bias conversion is the same BF16-bit expansion as the original.
Metal uses `MTLMathModeSafe`, like the baseline; no precision substitution,
affine factoring, or altered contraction policy is requested. Keeping the
same expression and lane order does not itself prove compiled bit identity;
the deferred GPU checks report bit identity and enforce both relative
L2 <1e-4 and max absolute <0.01. The test oracle uses separately rounded
FP32 products/additions and a row-major FP32 accumulation, not the packed
kernel's SIMD reduction. CPU layout simulation additionally checks against
a higher-accuracy sum of the FP32 terms.

Vocabulary output offsets and both existing argmax kernels are unchanged.
Their lowest-index tie selection and whole-logit NaN/Inf rejection are
covered by deferred GPU controls.

## Ownership, lifetime and memory

`Qwen36Weights` explicitly loads `embed_tokens` and `lm_head` into separate
`QuantMatrix` allocations; it does not tie or deduplicate them. `GModel`
copies descriptors and retains the original Metal objects/residency leases,
not the complete weight data. The implementation never replaces or mutates
either original allocation. A native `sourceAlias` field reports whether all
three embedding/classifier buffer objects actually alias, including callers
that explicitly supply tied descriptors.

Batched prefill and split-suffix classification continue to bind the original
classifier buffers. Embedding lookup remains original. Packed storage cannot
replace those row-major accesses without changing their kernels, which is out
of scope. Tokenwise graph `Forward` classification uses the selected decode
head, including tokenwise prefill when batched prefill is disabled.

Off mode creates **zero additional GPU buffers, zero candidate pipeline and
zero packing work**. On mode adds one 540344320-byte shared Metal buffer
(515.3125 MiB), no GPU scratch and no CPU repack staging allocation. Packing
writes directly from the three retained checkpoint buffers into that buffer.
Thus the additional transient payload peak equals the persistent additional
payload: **540344320 bytes**. Original classifier plus candidate representation
is **1080688640 bytes**; the separate, already-existing embedding is not an
experiment allocation. The small stack temporaries are not GPU scratch.

Before allocation, `LMHEAD_PREFLIGHT` prints payload bytes, current device
allocation, projected payload-based allocation and zero repack scratch.
`LMHEAD` prints actual `MTLBuffer.allocatedSize`, actual additional transient
allocation peak, requested/effective mode, geometry, layout version, alias
status and `batchedPrefill=original`. Allocator padding, if any, is included
in the actual allocation fields, not guessed from payload length. The
`di_graph_lmhead_info` six-u64 ABI returns mode, payload length, allocated
bytes, additional transient allocation peak, GPU scratch bytes and alias
status. These are representation accounting, **not** whole-process,
physical-residency or full-model high-water measurements. Existing graph
stats include the extra Metal allocation automatically.

When residency is enabled, the candidate joins the same bounded residency
manager under its own shared lease; budget failure is explicit. Graph reset
retains immutable packed weights and mode. Descriptor copies retain the
lease; the last owner releases residency and buffer storage on destruction,
including graph-construction failure. Selection is immutable, before any
layer registration, prefill preparation or execution. It cannot change at
reset. No service or launch configuration is modified.

## Selector

For the managed native-graph executor, launch the unchanged command with:

```sh
DEEPINFER_METAL_LMHEAD=off <existing-command-and-arguments>
DEEPINFER_METAL_LMHEAD=tile4 <existing-command-and-arguments>
```

Unset is off. Every other string, including an empty value, is an error.
A candidate with unsupported geometry/format
is rejected, not silently mapped to baseline. The single-model loader also
rejects candidate selection when SERIAL, EXPERT_UNFUSED, GENERIC or
HOST_EXECUTOR settings would bypass its native graph. Native-only callers
use `di_graph_lmhead_configure(graph, 0|1, error, capacity)` once, before
registering layers; there is no separate native environment selector.
An off native graph remains compatible with existing unconfigured callers.

The evaluation-harness preparation found that `SplitService.Profile` cleared
this managed selector before model construction. Its environment reset now
preserves only `DEEPINFER_METAL_LMHEAD`; all other managed experimental settings
retain their original reset behavior. This is required for `split-serve`
candidate-on to reach the existing selector, not kernel retuning. The parent
must freeze and package this correction before evaluation.

## Prepared evaluation harness (not executed)

The [harness README](../../../scripts/README-lmhead-evaluation.md) supplies
concrete read-only preflight and explicitly exclusive execution commands,
phase dependencies, fresh parent High Power evidence schema, all input/tool
pins, recovery behavior and resource/metric limitations.

`scripts/run-lmhead-evaluation.py` orchestrates gates/fixed/pilot/cohort without
building packages. `observe-lmhead-e2e.py` preserves two identical observation
hops for every arm and labels unchanged `ServiceEvent.result.timings`.
`summarize-lmhead-evaluation.py` reconciles clocks/tokens against the pinned
independent standard summarizer, retains failures/null metrics and reports
predeclared comparisons without automatic adoption. `issue103_common.py`
contains shared safety/metrics primitives; `test-lmhead-evaluation.py` exercises
them with CPU-only synthetic/mock evidence, including the independent pinned
summarizer on synthetic events. New output directories never overwrite runs.
Private configuration and request evidence stay in ignored restricted output.

No harness preflight/execution, GPU test, inference, Docker container, service
operation or power change has run during this preparation. The streaming API
does not expose full checkpoint logits/internal states: no such oracle is
claimed. Independent native operator checks and exact full-model sequences,
request reset/natural EOS/cancellation recovery remain mandatory scheduled
gates. Historical input-token-ID manifests are pinned, but only actual input
counts can be rechecked through this immutable API. Continuous physical
residency/device high-water queries are unavailable; sampled device allocation,
RSS and memory pressure are explicitly distinguished.

Preparation validation commands (full raw logs remain in the ignored evidence
root, with separate attempts):

```sh
python3 scripts/test-lmhead-evaluation.py -v
python3 -m py_compile scripts/issue103_common.py scripts/*lmhead*.py
python3 scripts/run-lmhead-evaluation.py --help
python3 scripts/observe-lmhead-e2e.py --help
python3 scripts/summarize-lmhead-evaluation.py --help
dotnet build DeepInfer.slnx --no-restore
git diff --check
```

The final harness CPU/synthetic suite passes **32/32**, including agreement
with the pinned independent summarizer. Python compilation/CLI help and diff
checks pass. The solution builds with zero warnings/errors. Complete logs and
exit codes are retained in `harness-implementation-final-02/`,
`harness-cli-final-02/` and `harness-build-02/` under the same local-only evidence
root; earlier attempts remain intact. These checks do not establish live
service recovery, GPU correctness, resource acceptance or measured speed.

## Validation and raw evidence

All commands below are run from the repository root. Every script invocation
requires a new output directory, saves the complete stdout/stderr, command,
exit status, source SHA/dirty patch, relevant file hashes and log hashes, and
does not overwrite previous attempts. Raw evidence is local-only and ignored
by Git under `artifacts/dual-host/issue103-lmhead-affine8/`.

CPU and compilation commands (no GPU dispatch):

```sh
bash scripts/check-metal-lmhead.sh --cpu artifacts/dual-host/issue103-lmhead-affine8/implementation-cpu-01
bash scripts/check-metal-lmhead.sh --compile artifacts/dual-host/issue103-lmhead-affine8/implementation-compile-01
dotnet build DeepInfer.slnx --no-restore
dotnet restore DeepInfer.slnx --locked-mode
dotnet build DeepInfer.slnx --no-restore
bash scripts/check-metal-lmhead.sh --cpu artifacts/dual-host/issue103-lmhead-affine8/implementation-cpu-02
bash scripts/check-metal-lmhead.sh --compile artifacts/dual-host/issue103-lmhead-affine8/implementation-compile-02
dotnet test tests/DeepInfer.Tests/DeepInfer.Tests.csproj --no-build --no-restore --filter 'FullyQualifiedName~LmHeadSelectorTests|FullyQualifiedName~ModelTests'
bash scripts/check-metal-lmhead.sh --cpu artifacts/dual-host/issue103-lmhead-affine8/implementation-cpu-final
bash scripts/check-metal-lmhead.sh --compile artifacts/dual-host/issue103-lmhead-affine8/implementation-compile-final
bash scripts/check-metal-lmhead.sh --compile artifacts/dual-host/issue103-lmhead-affine8/implementation-compile-packaging
dotnet build DeepInfer.slnx --no-restore
python3 scripts/build-progress-site.py
python3 scripts/test-home-trends.py
bash -n scripts/check-metal-lmhead.sh
git diff --check
```

Initial CPU and compile checks passed. The first managed build failed solely
because the fresh checkout had no NuGet assets; the locked restore and
subsequent build passed with zero warnings/errors. Both attempts remain in
`implementation-managed-01/`. The `-02` CPU check passed the full real-size
roundtrip (all codes and BF16 bit patterns), row-tail/sentinel/invalid-size
checks and five independent CPU FP32 operator cases; the largest max
absolute error at that revision was 0.0000343322754. The `-02` driver,
bridge and Metal library/pipeline compile passed without GPU submissions.
The final CPU check additionally includes larger signed affine parameters:
all cases pass; its largest max absolute error is 0.000671386719. The
managed CPU-only selector/model selection passes 12/12 tests, including
invalid/empty selectors and every host-baseline bypass guard, without
constructing a Metal device. Final compilation and managed build pass.
The notebook/site build and 26 offline site tests pass, as do shell syntax
and diff whitespace checks. The final packaging compilation, including
recorded driver/library hashes, passes. All attempts remain in the same
evidence root with distinct log/run names.

Final raw-log SHA-256 values (local-only paths relative to that root):

| Log | SHA-256 |
|---|---|
| `implementation-cpu-final/cpu-checks.log` | `840b32f67587a7a3bb0c1076959bf8ed6833c6506a7ee3f8c869b86c71403ac0` |
| `implementation-compile-packaging/metal-compile.log` | `a8f29bc3fe7df664e2ef823e79a8f75cf55f427aa85bb937a802f756b3080d81` |
| `implementation-managed-01/cpu-tests.log` | `72497171ba1afa84c7a24ff79f9d90be6a3d4e6596f5efb304412aa181270436` |

**Deferred until the parent provides exclusive GPU scheduling:**

```sh
bash scripts/check-metal-lmhead.sh --gpu artifacts/dual-host/issue103-lmhead-affine8/scheduled-gpu-01
bash scripts/check-metal-lmhead.sh --bench artifacts/dual-host/issue103-lmhead-affine8/scheduled-bench-01
```

The GPU driver fixes original vector quantization and row tile 4, and checks
small operator tails (N=1/3/4/5/15/16/17/33/65, all K=2048), zero/signed-zero
and signed scale/bias, cancellation/large inputs, group-boundary sparse inputs,
determinism, sentinels, original versus candidate, independent CPU reference,
real N=248320 logits, unchanged argmax, invalid modes/shapes, off allocation,
reset and shared-lease destruction. GPU correctness enables existing
residency; its benchmark disables residency in both arms. No full model
is loaded by either driver.

The bounded synthetic real-dimension microbenchmark performs 2 excluded
warmups and 5 formal samples **per arm and cache condition**; order alternates
AB/BA. Warm has no explicit eviction. Pressure reads a separate 134217728-byte
buffer before **each** sample, in a separately timed/completed command, with
4096 bytes of pressure-output scratch. These are harness-only allocations,
not runtime candidate scratch. Every sample is exactly one GEMV dispatch;
GPU command time excludes packing, CPU readback/reference checks and cache
pressure. Pressure time is separately logged. All warmup/formal timings,
geometry, source/candidate/harness bytes, mode/layout version, and correctness
checks are retained. The synthetic pattern is not claimed to represent
checkpoint logit distributions or end-to-end decode. For a different bounded
run, invoke the logged `lmhead-driver --bench W T P`, where W=1..5, T=1..10
and P=64..256 MiB; retain stdout/stderr and the binary hash.

## Remaining risks and handoff

GPU numerical correctness, bit identity, generated code/register pressure,
cache behavior, real checkpoint logits, prefill equivalence, full-model
reset/cancellation/natural EOS, loading/steady-state memory pressure and
HumanEval5/20 are untested. The one extra LM-head representation may worsen
working-set pressure; input reuse does not establish a bandwidth or decode
gain. The compilation-only phase cannot establish acceptance or adoption.
Freeze only after scheduled independent checks; retain failures, do not tune
on HumanEval20, and do not plot a trend point before qualified measurements.
