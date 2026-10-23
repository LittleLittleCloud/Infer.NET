# BF16 parity investigation

This is an opt-in diagnostic experiment, not an accepted deployment profile. The fixed numerical gates in QWEN36-8BIT-TARGETS.md remain unchanged.

## Observed rounding amplification

On source `3fee48879cd5b92c8525a0910fdc04a5b61095f5`, an 84-token, 8-output split verification with affine8 weights, FP32 GEMM and BF16 KV failed 280 of 1748 checks. Maximum relative L2 was 0.0023134101618249546; maximum absolute difference was 0.0625. The same prompt with FP32 KV passed all 1748 checks. This BF16-storage experiment is distinct from the required full BF16 GEMM profile.

For both backends, every layer3 BF16 key/value element exactly matched round-to-nearest-even applied to that backend's retained FP32 state. At position1, V feature331, Metal reference -0.04919430613517761 and CUDA -0.04919445514678955 differ by 1.4901161193847656e-7. They round to -0.049072265625 and -0.04931640625, a 0.000244140625 gap. Thus a correct BF16 conversion can magnify a pre-existing arithmetic discrepancy. This observation does not prove that every later discrepancy has the same cause.

Raw local evidence: `artifacts/dual-host/bf16-storage-check-3fee488-legacy` and `artifacts/dual-host/decode-check-3fee488-legacy`; rounding examples are in the first directory's `rounding-boundaries.json`. Raw artifacts remain local.

## Ordered reduction hypothesis — not yet validated

`DEEPINFER_METAL_MATH=ordered` replaces Metal shader simd_sum calls with explicit XOR shuffle additions in offsets16,8,4,2,1, matching CUDA sum32. Unset uses the existing Metal built-in. No tolerances, quantizer, weights or default deployment arithmetic change. This does not align transcendental functions, fusion, or every block-level reduction.

Validation must first run native graph/operator numerical checks, then compare whole-model FP32 and BF16 states against the retained baseline and the CUDA output on the same source commit. Any improvement must still pass the original gates; fewer failures alone is not acceptance.
