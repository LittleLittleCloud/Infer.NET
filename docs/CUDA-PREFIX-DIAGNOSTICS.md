# CUDA prefix profile and rejected BF16 experiment — 2026-10-23

## Measured bottleneck

An actual fixed8192-token prompt on the pinned affine8/group64 model took **13.132s** in the native CUDA front20layers without instrumentation; the event-instrumented run took **13.233s** (+0.77%). One warmup, one unprofiled run and one profiled run were used for diagnosis, not a formal throughput cohort. Hidden output and all40final states are byte-identical across the three runs.

Source `46679c40ad49a9d4f4993ddd5ce3657e955597a1`; FP32 GEMM/KV,32-token layer-major chunks, RTX4090, no competing model process. The fixed token-ID JSON SHA256 is `e52710e57e42391d4330032372c8641ecd61b620bba4cf2be9b515b1796a75ce`; pinned prefix weights SHA256 `ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77`. Model revision and input prompt hash remain those in the fixed target document.

| Phase | CUDA event interval, ms |
|---|---:|
| norm_and_input_projections | 4456.732 |
| delta_convolution | 137.877 |
| attention | 942.950 |
| output_projection_norm | 1587.427 |
| router_shared_gate_up | 969.267 |
| expert_gate_up | 3161.452 |
| expert_down_shared_down | 1844.176 |
| merge | 27.364 |
| hidden_readback | 38.095 |

Dense/input/output projections and router/shared work dominate together, followed by routed experts. Attention is about0.94s at8K. Event intervals can include host dispatch gaps; the first phase also includes input upload/embedding. These are not hardware-counter kernel-only timings. The probe excludes network, Metal suffix, incremental KV export and decode. It cannot replace end-to-end throughput or predict128K by linear scaling.

Reproduce with `scripts/split/profile-prefix.py` at the recorded source, an idle GPU, the pinned prefix file and fixed token-ID JSON. Original phase logs and binary hashes are retained locally under `cuda-prefix-profile-46679c4-retry`. An earlier transfer attempt never began GPU measurement and remains marked INVALID.

## BF16: a failed hypothesis retained

The default FP32 cross-host check still passes1748tensor/logit checks. Merely storing KV in BF16 causes280failures in the84input/8output case. Both backends implement the correct round-to-nearest-even conversion of their own FP32 state: a tiny pre-rounding difference can straddle a BF16 midpoint and amplify through later layers.

An opt-in Metal reduction order experiment attempted to match CUDA's shuffle summation. It **did not fix the error**: BF16 failures increased to369; maximum relativeL2 rose from0.0023134 to0.0658194, maximum absolute error from0.0625 to0.1549683. All8greedy IDs still agreed, which demonstrates why matching a short answer is insufficient. Default FP32 and ordered FP32 both passed, with maximum relativeL2 about1.50e-5 and1.57e-5 respectively.

Source `421dd763a47f0443533ac6d913fa4d7cba59d6ad`, native math option `DEEPINFER_METAL_MATH=ordered`, off by default. Both native/AOT builds and existing test suites passed, but the full cross-host BF16 numerical gate failed. The experiment is rejected, not deployed. The reference, tolerances and8bit weights are unchanged. These runs used FP32 GEMM and compared KV storage formats; they do not validate the formal BF16 GEMM execution profile.

## CUDA attention operator, not a model speedup

A separate partitioned attention kernel passed independent double-oracle cases and retained CUDA attention comparisons, including causal tails and FP32/BF16 KV. At131072keys with32queries, median FP32 operator time was84.16→21.25ms (128-key tiles); BF16 KV was23.74→16.79ms (256-key tiles). Smaller8K/32K benefits are much smaller, and tiny inputs can regress. The partial-output workspace must be included in any integration memory budget.

Source `55afd1049be9e8d4f55a8a863c9cdd2159139b90`, `scripts/windows/check-prefill-attention.ps1`, one warmup plus five operator trials, hot data. This kernel is **not integrated into the deployed executor**; these numbers are not full-model tokens/s. Public records are in `site/prefix-diagnostics.json`, with raw logs retained locally.

Dense weight reuse subsequently reduced deployed 32K prefill by 20.9%; see [the exact A/B record](CUDA-DENSE-REUSE.md). Formal BF16/131072-input gates remain open; no threshold has been relaxed. See [Issue3](https://github.com/LittleLittleCloud/Infer.NET/issues/3).


## Whole-prefix attention integration: FP32 passes, BF16 still fails

The experimental integration at `092e7ff0bd8b41d71fc40150157b7ff32b6e2469` tested 131 fixed token IDs in irregular 1/7/17/32-token chunks. Partitioned attention was forced on every chunk and compared with the retained attention path, on the same RTX 4090 and the same pinned weights. Both paths use packed FP32 GEMM. All 2,620 per-token/layer hidden checks and 40 final recurrent/KV states were tested against unchanged relativeL2 < 1e-4 and maxAbs < 0.01 limits.

- FP32 KV: **2,660/2,660 pass**, worst relativeL2 5.88494e-7.
- BF16 KV: **15/2,660 fail**, worst relativeL2 0.0400159 at token45/layer19 (maxAbs 0.00936964). The first failure is token24/layer15.

This isolates a sensitivity to changing attention reduction/normalization order even on one CUDA device. It does not prove a router selection changed: this experiment did not record routes. Passing the standalone operator and matching a few output IDs are insufficient for whole-model numerical acceptance. The BF16 candidate remains rejected; thresholds, reference and weight quantization were not changed. This is FP32 GEMM with BF16 KV, not the formal BF16 GEMM profile. No 128K whole-model throughput claim is made.

The experimental branch is not deployed. The accepted `8b5e3bd` packed FP32 deployment was restored and its real split request matched all eight baseline output IDs. Detailed failed tensors and source/binary identity are retained in `site/prefix-diagnostics.json`; full local results are `artifacts/dual-host/cuda-long-attention-092e7ff/result-kv{4,2}.json`.


## 32K whole CUDA prefix: partitioned attention rejected

The follow-up at `381b29c6921c3c5a01e49f279e04dd9bd0d0b013` used the real fixed32768-token prompt (token-ID file SHA256 `4248176bd03278693b6573f6bb63990d65f773259d8dde63256cbcd6b22ba50b`). Pinned affine8 weights, FP32 GEMM/KV, packed expert reads, CUDA32 chunks and the same native DLL (`a4de81bab49c171ce8a74f4be0b241a94810474a830de9a178599241472cbe6b`). The retained path received one warmup plus three measured runs: **49.716 / 50.260 / 49.783 / 49.917s**, formal median **49.917s** for the native CUDA front20 layers only.

The partitioned path switched at8192 keys. Its one warmup took **48.946s**, but **11,176 of 32,808** output/state checks failed the unchanged relativeL2 <1e-4 and maxAbs <0.01 gates. First failure was hidden token14975 (relativeL2 **0.027354**); worst relativeL2 **0.261746** at token28883, and a final state differed by up to **1.50058**. Of the failures, 11,163 were token hidden vectors and13 were final states. The measured candidate cohort was stopped immediately, so the warmup is **not** a speedup estimate. Windows whole-GPU sampled peak was **20,211MiB**. The native prefix excludes network, Metal suffix, state import and decode.

The experimental code remains off; the accepted `73c3d5d` service with packed FP32 CUDA prefill and128-token Metal scratch was restored. A real8K split request matched all eight retained output IDs with the same Mac PID. The first failure of the fixed numerical gate occurs well after the8192-key switch, consistent with error amplification; without a route trace, a specific expert-routing change is unproven. This experiment also does not validate BF16 GEMM/KV or the formal131072-token targets. Raw results and per-check failures remain local under `artifacts/dual-host/attention-scale-32k-20261023T153934Z`; the compact public record is in `site/prefix-diagnostics.json`.


### First failing layer at token14975

A focused replay of the same fixed prompt traced all20 CUDA-prefix layers for tokens14944–14975 on source `acbb630`. The **first check exceeding the original numerical gate is token14975/layer17**, relativeL2 **0.0320165** and maxAbs **0.00855136**. The preceding layer16 is relativeL2 **6.22e-6** (maxAbs **3.82e-6**); layers18/19 remain around2.74e-2. Slight, passing differences are visible earlier in the chunk. This narrows the abrupt amplification to layer17 but does not identify the operation inside it. The next probe records its expert IDs and router scores; a routing change remains unproven until that readout succeeds. The accepted service was restored after the isolated replay. The compact layer17 evidence is published in `site/prefix-diagnostics.json`; full640 layer comparisons remain in the local `attention-breakpoint-20261023T155022Z` artifact.
