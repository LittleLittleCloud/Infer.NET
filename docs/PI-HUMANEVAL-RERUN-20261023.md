# Pi × HumanEval Fixed 20 Problem Re-test: 19/20, Tool Call Format Failure

Date: 2026-10-23 (America/Los_Angeles). Related to [Tool Call Issue #32](https://github.com/LittleLittleCloud/Infer.NET/issues/32) and [Evaluation Tool PR #42](https://github.com/LittleLittleCloud/Infer.NET/pull/42). The coordinator, Mac split-serve, and Windows split-agent are from `e017d84e805c016d5912b5051e4c69f758056363`; the newly merged `15710714e537372a59b3c1ba999f31337a37017d` provides only the evaluation script. It is not a performance measurement of a new inference engine.

## Results

| Metric | This Round |
|---|---:|
| Pi Tasks Completed / Total | 19 / 20 |
| Syntactically Valid Solutions | 19 / 20 |
| Passed Official HumanEval `test` Field | **19 / 20** |
| Timeout | 0 |
| Total Serial Wall Time | 774.288 seconds |
| Median Wall Time per Problem / P90 | 31.798 / 64.703 seconds |
| Model Requests | 103 |
| Pi Report Input / Output Tokens | 241162 / 19949 |

Failed problem `HumanEval/149` returned `stopReason=error`, `errorMessage=terminated` in approximately 9.039 seconds on the first model request, with no `solution.py`. Meanwhile, the coordinator log recorded `System.IO.InvalidDataException: Malformed model tool call`, with the stack trace at `DeepInfer.Runtime.ChatTools.ParseOutput` → `DeepInfer.Host.CompatibleChat.Writer.End`. Retrying the same problem individually again failed with the same exception, taking 9.068 seconds. Therefore, 19/20 in this round is the actual end-to-end result, not a case where the code answer was asserted as incorrect; individual retries were **not** included in the 20 problem statistics. The service subsequently remained Ready, agent connected, and the remaining problems completed normally.

The failed problem ended prematurely, so the total wall time cannot be used to infer worker prefill/decode speed. Per-problem data is available in the [public summary JSON](../site/pi-humaneval-rerun-summary.json). This is still a small sample of Pi coding tasks with tools for the fixed 20 problem, **not** the official leaderboard pass@1 for the complete 164 problem set, nor is it an acceptance test for 128K prefill/decode.

## Configuration and Verification

- Dataset: OpenAI HumanEval revision `b9453e8f1865df0a0057a1383767caa400acfbcf`, compressed file SHA-256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`; seed 0, fixed identical 20 problems and order. Pi 1.4.0, independent working directory and session per problem, 300 seconds per-problem timeout, 32768 context window, 4096 single output limit, `read/write/bash/edit` tools.
- Model: Qwen3.6-35B-A3B revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine 8bit weights, **FP32 KV**. Mac M5 Pro 48 GiB full model and decode, RTX 4090 resident for the first 20 layers; cross-machine via Wi-Fi. Mac plugged in, High Power. Before evaluation, the worker was Ready, agent connected; Windows GPU was idle; the service remained healthy after completion. The user had the TUI open simultaneously, so it cannot be claimed that the environment was fully exclusive.
- `scripts/benchmark-pi-humaneval.py` in `main`'s SHA-256 was `594f60571fabc5d05d7aa849cfd25ad668a3ee8fed7c5f278b6a7ea47b9d9c22`. Local offline fixture 12 items passed, 2 items skipped (Windows-specific); after merging, Linux, Windows, and macOS tasks in GitHub CI all passed.
- Complete `solution.py` was run in Docker containers with no network, read-only, no extra capabilities, limited CPU/memory/PID, and non-root, executing the corresponding official `test` and `check(entry_point)`; all 19 instances passed. Intentionally incorrect negative controls failed. The scoring script is [`scripts/eval-pi-humaneval-docker.py`](../scripts/eval-pi-humaneval-docker.py), and the container image is fixed at `python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd`. Model-generated files are not executed on the host.

Reproduction command (from the repository root, using a private token file and the same-version service started in an isolated local environment):

```sh
python3 scripts/benchmark-pi-humaneval.py \
  --cohort cohort --seed 0 --timeout 300 --context-window 32768 --max-tokens 4096 \
  --dataset /PRIVATE/HumanEval.jsonl.gz --token-file /PRIVATE/token.txt \
  --endpoint-build e017d84e805c016d5912b5051e4c69f758056363 \
  --model-revision e06a74e6236a60c8367e1a3214e83d8b61b637b0 \
  --output /NEW/IGNORED/RESULT_DIRECTORY

python3 scripts/eval-pi-humaneval-docker.py \
  --results /RESULT_DIRECTORY --dataset /PRIVATE/HumanEval.jsonl.gz \
  --output /RESULT_DIRECTORY/functional-tests.json
```

Original events, generated files, and full scoring logs are saved in `artifacts/humaneval-main-rerun-20261023/`, which is ignored locally; single-question reproduction is saved in `artifacts/humaneval-149-retry-20261023/` and is not available for public download. SHA-256: This round's `manifest.json` `d7c01e21a28780b5f4a4d27b23826908f757fbd9a06f943d6b90a47a15984aad`, `tasks.jsonl` `7080013309968687a167413378f397e9357fc9a1815be3fbafabecdb1344277b`, `summary.json` `b4049089b744136dc06f777e71fe34e35cabc45f86595ac2b3e056c48d97617e`, `functional-tests.json` `0cb18ffb96d20a2a339765d5b9c1fb9e16fef356eae67773ec95f1f5851656b4`; single-question retry `tasks.jsonl` `a2dff920fcf109dc961976ee8a50b769d885f87762e8b637b39473150ff0d57c`.

The next step is to identify the model output formats that `ChatTools.ParseOutput` cannot accept, and add regression coverage for real model outputs and error responses. After the fix, the same 20 question should be rerun completely, rather than replacing failed samples with single-question retries.
