# Experimental native DFlash2

[Issue #80](https://github.com/LittleLittleCloud/Infer.NET/issues/80) adds an optional local Metal DFlash2 drafter for the existing Qwen3.6 affine8 target. It is a candidate implementation, disabled by default. Build success is not numerical or performance acceptance; see the [experiment record](EXPERIMENT-80-DFLASH2-20261024.md) for actual validation status.

## Weights and precision

Use `incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b6923ad6c14cb1bb41c37a9041446496ab6`, with the unmodified pinned `config.json` and `model.safetensors` in one directory. The loader verifies both file hashes, the complete tensor set, shapes, and BF16 dtype. Weight SHA256: `d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d`; config SHA256: `86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59`. Download from the pinned model revision and retain the model license with any redistributed weights. Weights are not checked into this repository.

The six-layer draft retains BF16 matrices and uses FP32 arithmetic, activations, accumulation, and context KV. Draft RMS scales are direct trained BF16 values converted to FP32; do not apply the target/MTP raw-plus-one conversion. The target embedding and vocabulary head are shared, retaining their original affine8/group64 configuration and tokenizer. There is no CPU weight-compute fallback, draft quantization, FP16 staging, or alternative serving backend.

The algorithm reference is `z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a` (`dflash/model_mlx.py` and `dflash/model.py`, MIT). This implementation uses its own C#/Metal code and the checkpoint's dynamic convolutions and trained candidate selector. Minimal exact verifier, state transaction, and allocation tracing infrastructure was explicitly ported from the unmerged Issue #76 implementation `17965ce1f43e37398441c6f192c0a331efab228d`; this does not imply PR #78 was adopted.

## Running a candidate

Build and package through the existing deployment workflow, then add these options to a separate worker configuration/package:

```sh
deepinfer split-serve --config PRIVATE_CANDIDATE_WORKER_CONFIG \
  --dflash2 8 --dflash2-model PINNED_DRAFT_DIRECTORY \
  --decode-attention fused-fp32 --kv fp32 \
  --metal-scratch-tokens 128 --suffix-chunk-tokens 128 \
  --short-prefill-tokens 262144 --metal-residency on
```

Use `--dflash2 4` for an anchor plus three proposals, or `8` for an anchor plus seven. Remaining-output tails use smaller actual draft blocks or a single autoregressive step. `--dflash2 off` is the default and creates no draft weights, cache, verifier snapshots, or tap buffers. Model/replay options without an enabled mode are rejected.

Only greedy, FP32-KV, fully local Mac Metal execution is supported. The enabled worker advertises and enforces a 32768-token combined prompt/output window and 4096-token output cap; it rejects remaining-context output mode, unsupported options, and split prefill rather than changing paths. Existing authentication and queue/backpressure apply. Use separate worker IDs, locks, ports, and logs; never load another full model alongside the original resident service.

## Execution and state

Target prefill captures raw post-layer residuals at zero-based layers 1, 6, 11, 16, 22, 27, 32, 37, before the final target norm. The drafter projects their concatenation and builds each draft layer's context K/V. Sliding attention needs only the last 2047 committed context rows; earlier prompt rows still run through the full target. Absolute RoPE positions survive ring eviction. Proposal K/V is ephemeral, and proposal attention is bidirectional within its block.

Each draft layer includes grouped two-tap causal dynamic convolutions around attention and the gated MLP. The shared head computes proposal-position logits. GPU top-16 selection and the trained predecessor/hidden/successor edge selector trace one proposal path from the verified anchor.

The target verifies `[anchor, proposals]` in a multi-row pass. It accepts the matching prefix and emits the target correction/bonus as the new pending anchor. On rejection, an owned snapshot restores all-layer recurrent/convolution state and replays only the committed prefix; rejected KV suffixes remain inaccessible. Only committed target feature rows update persistent draft context. Final unused draft context may lag the terminal target, with its valid prefix explicitly recorded; a new request resets it. Terminal target state and pending-token correctness are independently tested.

`--dflash2-replay serial` is a diagnostic oracle, not an acceleration mode. Production comparisons use `batched` replay. CPU code coordinates the cycle and checks small final outputs; matrix, attention, MLP, and selector computation runs on Metal.

## Timing and resource evidence

Prompt feature capture and needed draft-context bootstrap belong to prefill and TTFT. Decode clocks include drafting, selector, target verification, context ingestion, snapshots, rollback/replay, and host/readback work. Validated block tokens share the actual block-ready timestamp; no intra-block spacing is invented. `DFLASH2_STATS` logs proposed/accepted counts, acceptance histogram, and component timers per request. These component timers do not replace the standard sum(N−1)/sum(worker decode seconds) metric.

`DFLASH2_EVIDENCE` records weight-residency registration and allocation/submission-boundary samples. Whole-device `currentAllocatedSize` already includes draft allocations; do not add the logical draft subtotal again. Samples are not a continuous Metal-driver high-water, and registration does not independently prove physical residency of every KV/workspace buffer. Missing resource evidence prevents full acceptance. No system memory-limit changes are included.

## Validation commands

Compile-only checks create libraries/pipelines but execute no GPU kernels:

```sh
scripts/check-metal-dflash-compile.sh
dotnet build
dotnet test tests/DeepInfer.Tests --filter 'FullyQualifiedName~Dflash2'
```

With GPU opt-in absent, full-model facts are skipped; ordinary repository tests may contain other small Metal operators. The real-artifact loader fact uses `DEEPINFER_DFLASH2_DIR` without GPU execution. For full-model gates, arrange an exclusive maintenance window and run one fact at a time:

```sh
DEEPINFER_DFLASH2_GPU=1 DEEPINFER_TARGET_DIR=PINNED_TARGET_DIRECTORY \
DEEPINFER_DFLASH2_DIR=PINNED_DRAFT_DIRECTORY \
DEEPINFER_METAL_KV=fp32 DEEPINFER_METAL_PREFILL_CHUNK=32 \
DEEPINFER_METAL_PREFILL_GEMM=fp32 \
dotnet test tests/DeepInfer.Tests --filter 'FullyQualifiedName=DeepInfer.Tests.Dflash2GpuTests.TEST_NAME'
```

Facts cover actual scalar target logits/taps/state versus multi-row verification, independent FP32 draft/embedding/head equations, ring eviction and poisoned invisible slots, fresh target-conditioned context after rejection, absolute positions, fixed output/state fixtures, EOS, cancellation/reset, and serial replay. Set `DEEPINFER_DFLASH2_FIXTURES=128,4096` for warm-one/formal-three fixed fixtures. A test being present or skipped is not a passing hardware result. Run fixed HumanEval20 separately for comparative application measurements, preserving failures and metric coverage.
