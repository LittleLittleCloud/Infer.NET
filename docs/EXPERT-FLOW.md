# Expert intermediate layout and dequantization experiment

The optional candidate writes the fused expert gate/up result directly to the
BF16 layout consumed by expert down. The original path writes FP32 values in
token/route order, gathers them into expert tiles, rounds to BF16, then performs
down. The direct path performs the same FP32 gate/up and SwiGLU arithmetic and
the same BF16 rounding, placing each result at its final packed address.

## Implementation

`DEEPINFER_METAL_PREFILL_EXPERT_FLOW=staged|direct` selects the flow at model
startup. Direct flow requires grouped, packed BF16 experts with hybrid GEMM.
It uses the existing expert activation allocation as its output; the packed
gate input remains separate to avoid read/write races between threadgroups.
Only the private expert activation buffer receives the new layout. Other
operator outputs and short-input fallback retain their original representation.

At the selected 2048-token model chunk, both paths allocate the same buffer
space. Smaller configured maximum chunks can need a larger activation buffer
to accommodate up to 256 partial expert tiles. Dimensions, tails and original
token/slot ordering are retained when down scatters its FP32 output.

Each full chunk removes 40 packing kernels and their encoder barriers, one per
layer. GPU command-buffer counts remain unchanged. For each layer, 2048 tokens
times eight routes times 512 activations saves a 32 MiB FP32 write and 32 MiB
FP32 read. This is logical tensor traffic, not a measured reduction in DRAM
transactions; cache behavior and padded tails also affect physical traffic.

Weights stay five-bit/group64 with the existing eight-bit router. Residuals,
normalization, DeltaNet state, routing and decode GEMV keep their prior FP32
behavior. No model/runtime substitution, prefix reuse or changed output budget
is involved.

## Dequantization pipeline experiment

`DEEPINFER_METAL_PREFILL_EXPERT_PIPELINE=serial|double-buffer` retains an explicit
diagnostic switch. The double-buffer version prepares the next weight tile in
the alternate threadgroup buffer and reduces two barriers per inner step to
one, after an initial barrier. Gate/up threadgroup weight storage rises from
8 to 16 KiB; down rises from 4 to 8 KiB. Actual compute/load overlap and hardware
occupancy have not been measured with GPU counters.

The first real-model 8K comparison found this version slower. A second isolated
experiment using fixed buffer addresses also regressed and was removed from
the product source; it remains under `artifacts/reference/expert-flow-static/`.
The selected pipeline remains **serial**. A cooperative BF16 tensor-store
experiment failed exact layout comparison and was also removed. None of these
rejected implementations is enabled by the launcher.

## Measurement and evidence

**The existing staged flow remains the default.** Direct output improved 8K
slightly but did not demonstrate a 32K benefit, including an immediate repeat.
The M5 launcher is unchanged. Direct mode remains available explicitly through
`DEEPINFER_METAL_PREFILL_EXPERT_FLOW=direct` on the hybrid/packed BF16 profile.

| Confirmation input | Staged prefill | Direct prefill | Direct time change |
|---|---:|---:|---:|
| 8192 | 3.642 s | 3.592 s | −1.38% |
| 32768 | 18.558 s | 18.769 s | +1.13% |
| 32768, immediate direct repeat | same 18.558 s control | 18.646 s | +0.47% |

Every entry averages three measured requests after three excluded full warmups.
The original direct 32K samples rose from 18.569 to 18.780 to 18.956 seconds;
that drift prompted the extra repeat. The repeat was 18.677 / 18.686 / 18.574
seconds. No stable 32K speedup is claimed. Confirmation decode throughput at
8K was 79.85 vs 80.07 tokens/s (staged/direct); at 32K, 70.71 vs 68.13, with
70.31 in the repeat. These measurements do not establish a decode improvement.

In the separate pilot round, 8K staged/direct/direct-plus-double-buffer means
were 3.447 / 3.429 / 3.693 seconds. Direct improved by 0.51%; adding double
buffering then made it 7.68% slower than direct serial (7.13% slower than the
old staged path). Do not compare pilot and confirmation absolute times as a
code-speedup claim: the rounds show substantial drift.

All 48 real-model warmup/measured requests produce identical 128-token outputs
for their input length. Kernel counts fall from 94926 to 94766 at 8K and
118710 to 118070 at 32K; command-buffer counts stay 131 and 143. Allocated
Metal buffer totals are unchanged in these measured 2048-token configurations.
All 190 power observations report battery power with the same power settings.

The user selected High Power while the machine was on battery. Every request
records power source and settings before/after, with periodic samples during
generation. The confirmation sweep rejects a changed power source or settings.
This records power configuration, not actual GPU clocks. The earlier AC
attention experiment is a separate round and is not the control for this change.

Both the pilot and confirmation use the same pinned Qwen3.6-35B-A3B five-bit
checkpoint, exact saved 8192/32768-token prompts, greedy fixed 128 outputs with
EOS ignored, three excluded full warmups and three measured requests. GPU jobs
run serially. The confirmation uses control then direct at 8K and reverses the
order at 32K. Between-round drift is retained rather than attributed to code.

`scripts/summarize-qwen36-expert-flow.py` verifies binary hashes, raw RPC hashes,
identical input and output IDs, actual kernel/command-buffer counts, unchanged
precision/allocation and observed power state. Its output is
`artifacts/baselines/qwen36-expert-flow-summary.json`.

The first-round binaries are preserved in `artifacts/native/expert-flow-first/`;
the predecessor binaries are in `artifacts/native/before-expert-flow/`.
The full-model reports use `qwen36-expert-flow-*-8k-highpower.json` and
`qwen36-expert-flow-{control,direct}-{8,32}k-confirm.json`.
The extra repeat is `qwen36-expert-flow-direct-32k-recheck.json`.

Native AOT publication succeeds. The final source passes 146 real GPU checks:
21 direct-layout/composition cases in each serial and double-buffer mode,
56 BF16 grouped/shared expert oracle cases, and 48 original FP32 expert cases.
Both flow suites have identical output hashes. All 68 unit tests and 25
process/RPC integration checks pass. Logs use `artifacts/expert-flow-final-*`.
The direct candidate also passes four real-model RPC acceptance checks:
distant-needle retrieval, native context-limit rejection, independent short
greedy IDs after long-cache use, and deterministic arithmetic after state reset.
These are in `artifacts/baselines/qwen36-expert-flow-acceptance.json`.
The default staged/serial worker is restored on port 5100 after validation.

```sh
# Stop the owned idle model first; the runner refuses existing services.
artifacts/baselines/mlx-env/bin/python scripts/sweep-qwen36-expert-flow.py
python3 scripts/summarize-qwen36-expert-flow.py

# Real GPU layout, full gate/down composition and fallback checks.
bash scripts/check-metal-expert-flow.sh
DEEPINFER_METAL_PREFILL_EXPERT_PIPELINE=double-buffer \
  artifacts/native-checks/expert-flow-checks
```

The summary additionally expects the preserved pilot artifacts from this run.
No new MLX benchmark, prefill parity, maximum-window quality or long-output
throughput conclusion is claimed by this experiment.
