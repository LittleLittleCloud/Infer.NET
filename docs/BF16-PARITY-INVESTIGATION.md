# BF16 parity investigation

This is an opt-in diagnostic experiment, not an accepted deployment profile. The fixed numerical gates in QWEN36-8BIT-TARGETS.md remain unchanged.

## Observed rounding amplification

On source `3fee48879cd5b92c8525a0910fdc04a5b61095f5`, an 84-token, 8-output split verification with affine8 weights, FP32 GEMM and BF16 KV failed 280 of 1748 checks. Maximum relative L2 was 0.0023134101618249546; maximum absolute difference was 0.0625. The same prompt with FP32 KV passed all 1748 checks. This BF16-storage experiment is distinct from the required full BF16 GEMM profile.

For both backends, every layer3 BF16 key/value element exactly matched round-to-nearest-even applied to that backend's retained FP32 state. At position1, V feature331, Metal reference -0.04919430613517761 and CUDA -0.04919445514678955 differ by 1.4901161193847656e-7. They round to -0.049072265625 and -0.04931640625, a 0.000244140625 gap. Thus a correct BF16 conversion can magnify a pre-existing arithmetic discrepancy. This observation does not prove that every later discrepancy has the same cause.

Raw local evidence: `artifacts/dual-host/bf16-storage-check-3fee488-legacy` and `artifacts/dual-host/decode-check-3fee488-legacy`; rounding examples are in the first directory's `rounding-boundaries.json`. Raw artifacts remain local.

## Ordered reduction hypothesis — not yet validated

`DEEPINFER_METAL_MATH=ordered` replaces Metal shader simd_sum calls with explicit XOR shuffle additions in offsets16,8,4,2,1, matching CUDA sum32. Unset uses the existing Metal built-in. No tolerances, quantizer, weights or default deployment arithmetic change. This does not align transcendental functions, fusion, or every block-level reduction.

Validation must first run native graph/operator numerical checks, then compare whole-model FP32 and BF16 states against the retained baseline and the CUDA output on the same source commit. Any improvement must still pass the original gates; fewer failures alone is not acceptance.

## Ordered reduction A/B, 2026-10-23

The diagnostic was cherry-picked onto the BF16 prefix branch at `2e064a7`, with no CUDA source changes. The fixed CUDA trace is `bf16-cuda-probe-8408223` (`fc52acaa68f9205277124f8871bfec417277e5425961755ce327c34198800582`); the later `92f24b5` value probe confirmed that its trace was unchanged. Both Metal runs use the same pinned affine8 model, synthetic 128-token IDs, BF16 KV/GEMM, FP32 diagnostic router, 128-token prefill chunk, matrix path and sequential Delta. The only Metal execution change is `DEEPINFER_METAL_MATH=ordered`. The original trace was re-compared with the same script and CUDA trace, rather than relying on rounded figures in an earlier report.

| Metal reduction | Layer 0 relative L2 | Layer 1 relative L2 | Passing trace/state checks | Worst state relative L2 |
|---|---:|---:|---:|---:|
| Default `simd_sum` | 0.000160545 | 0.004015178 | 2 / 60 | 0.0711569997 |
| Explicit XOR order | 0.000160687 | 0.004015629 | 2 / 60 | 0.0624690843 |

The first failing layer is unchanged to the precision that matters for the fixed `<1e-4` relative-L2 gate; ordered reduction does **not** fix BF16 parity. Some later states improve, while `state-12-1` worsens by about 0.01014 relative L2, so the change is not a monotonic numerical improvement. Both traces report the same next token (`16`), which does not establish state or long-sequence correctness. This A/B does not measure throughput, long context, transfer, or full decode, and is not a production profile.

The ordered run passed native graph checks (60 hardware checks plus scratch/profile checks), BF16 dense/expert checks, and 69 fused-attention checks. The managed Host build and Metal dylib build passed. Raw evidence remains local under ignored `artifacts/dual-host/`: `bf16-metal-ordered-2e064a7/trace.f32` SHA-256 `66c852ec5ef69c62954310dcc140fea4d1c0614f19893bc4940a5486286f9af4`, ordered comparison `bf16-ordered-compare-2e064a7.json` SHA-256 `2bb93a93af91a441b47588dcdd072ff6907cf5ae174a505b8ebf4d5bf3b146a1`, baseline re-comparison `bf16-baseline-recompare-2e064a7.json` SHA-256 `59d3f284df3773c26b8e49d0646f2a814a7c56a98e016e5e2713fa95cd91a661`. The ordered Metal dylib SHA-256 is `cdff47f073d3b726c12cf097790909dea911a67f970c0b77be5a638e2f669b30`. A CUDA rebuild was unnecessary for this isolated Metal-only switch; a final cross-host acceptance still requires two same-source platform builds and the full gates.
