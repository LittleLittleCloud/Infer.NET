# Official Qwen3.6 MTP experiment

Issue [#76](https://github.com/LittleLittleCloud/Infer.NET/issues/76); baseline `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3`. The human approved implementation and fixed HumanEval20 evaluation. This document describes candidate code, not an adopted optimization or measured speedup. DFlash is out of scope.

## Weights and execution

Target: `mlx-community/Qwen3.6-35B-A3B-8bit@e06a74e6236a60c8367e1a3214e83d8b61b637b0`, unchanged affine8/group64 with checkpoint overrides. Keep its tokenizer and share its embedding and output-head buffers.

Draft: official `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`, one full-attention/MoE MTP layer, 844,640,768 parameters. The converter authenticates complete official shards 25/26 (6,064,305,104 bytes total), extracts 19 tensors and splits the expert gate/up tensor. Matrices remain bit-preserved BF16; seven norm scales become FP32 `rawBF16.float()+1`. Runtime weight payload: 1,689,303,040 bytes before alignment and workspace. No full BF16 base model, draft quantization, FP16 staging or CPU serving backend.

```sh
python3 scripts/convert-qwen36-mtp.py --source-dir SOURCE_SHARDS --output-dir MTP_SIDECAR
```

The sidecar manifest records source, converter, tensor and output hashes. The loader verifies the pinned model/revision, geometry, precision, provenance and payloads. The parent independently audited the converted matrices, all norm scales and expert gate/up roundtrip. An opt-in `Category=MtpArtifact` test loads the authenticated real sidecar without a GPU, checking the converter/loader contract (including the official expert down tensor name without a `.weight` suffix).

## Generation semantics

MTP input pairs the next token embedding with the target's post-final-norm hidden row. Full-prompt bootstrap uses bounded 32-row chunks and belongs to prefill/TTFT. The initial verified anchor is emitted once and remains unconsumed; its first MTP proposal is computed afterwards and belongs to decode.

Each cycle verifies `[anchor,d1..dk]`, with widths 1/2 and `k <= remaining-1`. It accepts the longest greedy-matching prefix, restores and replays the committed prefix on rejection, then emits accepted drafts plus the target correction/bonus. The new anchor remains unconsumed. Before another speculative cycle, MTP KV is rewritten using accepted **target** hidden rows; matching token IDs do not validate draft-generated hidden/KV. EOS and output caps skip unused future draft work. A final AR-only tail does not draft.

The native verifier processes rows through the layers together and exposes every row's hidden, logits and argmax. Verification uses decode attention and decode-order SwiGLU; the classifier reuses weights across rows. These choices require hardware G3 exactness checks. Serial replay is diagnostic, never a serial verifier presented as acceleration.

Width-2 feeds the preceding post-MTP-norm hidden into the reused layer. This follows pinned vLLM `4e591657157a021cab997f911d67f50c0d18a71e`: `qwen3_5_mtp.py` lines 187–191 and 281–294 return normalized hidden, and `llm_base_proposer.py` lines 753–762/1001–1014 feed that single returned hidden forward (`Qwen3_5MTP` is not among tuple-returning architectures).

## Opt-in serving

Append to the existing private `split-serve --config ...` command:

```sh
--mtp 1 --mtp-sidecar MTP_SIDECAR --decode-attention fused-fp32 \
--short-prefill-tokens 262144 --suffix-chunk-tokens 128 \
--metal-scratch-tokens 128 --metal-residency on --kv fp32
```

Use `--mtp 2` for the second pilot. Enabled requests are bounded to 32768 total context tokens and 4096 output tokens, greedy and fully local Metal. Unsupported requests fail explicitly. MTP is off by default and allocates no draft/verify/snapshot GPU buffers on that path. No Windows or formal128K acceptance is claimed.

## Validation and timing

Follow Issue #76 gates in order: G0 conversion/oracle; G1 disabled path; G2 draft operator/reference agreement; G3 verifier/state/rollback exactness; G4 deterministic fixed128/4096 and natural-EOS fixtures; G5 both fixed five-task pilots; G6 fixed main/off/selected-on HumanEval20. Preserve all failures. GPU tests share one nonparallel collection and must run with no other full-model worker loaded.

```sh
python3 -W error::ResourceWarning -m unittest tests.test_convert_qwen36_mtp
dotnet test tests/DeepInfer.Tests -c Release
bash scripts/check-metal-mtp-compile.sh
```

The ordinary unit suite includes small real Metal operator tests on physical macOS. Full-model tests require `DEEPINFER_MTP_GPU_TESTS=1`, target/sidecar directories and `--filter Category=MtpGpu`; they skip by default. Schedule G2 and G3 separately and serially. GPU exactness and performance remain unmeasured until their result records exist.

Worker decode throughput is `SUM(N-1)/SUM(T_last-T_first)`. Block tokens may share readiness timestamps; no invented intra-block spacing. Include first draft, provisional drafts, verification, acceptance/readback, cache reconditioning and rejection restore/replay. Also report prefill, complete client TTFT coverage, application wall time, correctness and resources. The proposed 1.20x adoption threshold is a criterion, not a forecast.

`MTP_STATS` records cycle counts/costs. `MTP_EVIDENCE` reports residency registration and candidate allocation-boundary/submission samples. Allocation tracing is configured through an explicit native ABI before loading enabled models; managed environment changes do not configure libc `getenv`. `DEEPINFER_METAL_ALLOC_TRACE=1` at process launch can trace candidate-off diagnostics too. These samples include tracked allocation/growth transitions but are **not a continuous Metal-driver peak** and are not equivalent to main's response-time allocation samples. Keep that limitation explicit; missing required peak evidence prevents adoption. Residency registration covers draft weights; it does not prove physical residency of all KV/workspace. Record swap-out and pressure independently.

Code and results are separate PRs; no automatic merge or production adoption.
