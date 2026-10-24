# Merged main Run: Pi fixed 20 questions, 2026-10-23

Related to [Issue #44](https://github.com/LittleLittleCloud/Infer.NET/issues/44). After user review, first merge the results [PR #47](https://github.com/LittleLittleCloud/Infer.NET/pull/47), then merge the homepage code [PR #46](https://github.com/LittleLittleCloud/Infer.NET/pull/46), and confirm successful Pages deployment. The user subsequently requested a real Run of the post-merge main.

## Actual Test Results

| Metric | This Round Actual Test | Coverage |
|---|---:|---:|
| Worker prefill | **531.199 token/s** | 92/92 requests |
| Worker decode | **65.085 token/s** | 92/92 requests |
| Client median TTFT for first valid output | **6986.970 ms** | 92/92 requests |
| Official per-problem function assertion pass rate | **100%, 20/20** | Fixed complete 20 problems |
| Pi full round end-to-end task duration | **711.369460043 s** | 20/20 problems |
| Cumulative input / generated tokens | **211383 / 19892** | 92 model requests |

Zero task timeouts, zero Pi execution errors, 20 out of `solution.py` all passed the corresponding HumanEval official assertions in isolated containers. Negative control with wrong answers actually failed with `AssertionError`, exit code 1; positive control exit code 0. Time per problem and cumulative token count are retained in the homepage task table; multi-round repeated context counts as input, not net prompt length.

Prefill uses `Σ inputTokens / Σ worker prefillSeconds`; decode uses `Σ(outputTokens−1) / Σ worker decodeSeconds`, not averaging token/s per request. Input/output counts per request and Pi's independent usage match item by item. Both speeds and TTFT reach the full round 92/92 collection coverage.

This is a real main baseline for Pi coding agent load, not a full 164 problem leaderboard result, nor 128K formal performance acceptance.

## Version, Configuration, and Observation Boundaries

Mac actually runs coordinator, split-serve, and Windows split-agent all rebuilt, deployed, and verified as the same complete main source code:

`cd6f7da401fadf044dac13e87b415d27bb5eadfb`

The Mac server PID remained 50216 during this round; the Windows actual running agent PID was 69196, and the new package path and version have been verified. The original package and configuration backups are retained. The two Mac components are managed by the original user's LaunchAgent; Windows continues to use the foreground agent from the retained SSH session, with no new SCM service or administrator configuration added. New package verification passed for Mac 9 items / Windows 10 items; the real deployment acceptance after stable readiness prior to formal evaluation passed 15 items. The tested source SHA is the actual binary version, not the future merge commit recorded in this result.

| Artifact | SHA-256 |
|---|---|
| Mac executable / Metal dylib | See `binarySha256` in public observations.json, consistent with package manifest |
| Windows executable | `76555a5f42392e8ff87bb8a59b8b8f794b41d4d063c10d1eefe293ee79636869` |
| Windows CUDA DLL | `30d789415cbd68e17fe329fd65dcf2d2c4aef7725666d8937e712e09b4e742eb` |

- Model Qwen3.6-35B-A3B; revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`; affine8/group64 weights, FP32 KV.
- Prefill for the first 20 layers of 4090, Mac handles the remaining 20 layers and full decode; CUDA `batch-fp32-packed`, chunk 32, suffix 128, Metal scratch 128, incremental KV, fused FP32 decode. The local routing threshold for short inputs is 0, and all formal requests were observed as `cuda-metal-split`.
- Mac plugged in, High Power; Windows stock configuration; Mac Wi-Fi / Windows Ethernet, original TLS data channel. No other GPU benchmarks were started in parallel; no claim of exclusive desktop environment or full device peak resource acceptance.
- Pi 1.4.0; same 20 question with fixed seed 0; independent session/directory per question; tools read/write/bash/edit; thinking off; single question 300 seconds, context 32768, single output limit 4096; natural EOS. Cache read/write are both 0.

The current main service does not have per-request `INFERENCE_METRIC` logs. This round uses two independent loopback forwarding observations, and the product's .NET/CUDA/Metal inference code has not changed:

`Pi → HTTP observer client → independent .NET coordinator at the same version → raw worker stream observation → actual split-serve → TLS → actual split-agent`

Worker prefill/decode values are taken directly from real `ServiceEvent.result.timings`; there is no polling of the Dashboard for recent requests to replace per-request records. Each model request is aligned via single-concurrency window, task start/end times, worker request ID, and usage. The collector does not store request bodies, response texts, credentials, or public copies of generated code.

TTFT is measured from when the observation client starts sending the prepared request to the real .NET coordinator until the first non-empty text or valid tool call content is received, using the same process monotonic clock. Role, heartbeat, and usage frames are excluded. Pi dispatch to the observer is outside this boundary; Pi's total wall time for the question includes the full chain. The current tool call API buffers output, so TTFT may be close to the completion time of that generation and cannot be interpreted as the worker's internal first token time.

New observation hops, packed/routing configurations differ from old history, so this round uses an independent comparable configuration; the difference between 711.37 seconds and the total time of previous rounds is not attributed to any specific code optimization. Old `e017d84` records and missing test statuses are fully retained.

## Evidence, Reproduction, and Page Acceptance

- [Public per-task, per-request, and phase data](evaluations/main-cd6f7da-20261023/observations.json): 92 real worker timings, client TTFT, 20 question time windows, official assertion results, artifacts, and local raw file hashes.
- [New Run page item-by-item verification](evaluations/main-cd6f7da-20261023/page-checks.json), [Desktop screenshots](evaluations/main-cd6f7da-20261023/home-desktop.png), [320px screenshots](evaluations/main-cd6f7da-20261023/home-mobile.png): New Run default selection, four measured values, 20 row task time/token count, missing test status of old configuration, and deep link reload all passed, with no script errors. Site build consists of 3 pages / 46 notes / 3 real runs, with 18 Python and 15 JavaScript checks passed.
- [Reproducible observer](evaluations/main-cd6f7da-20261023/observe-main-run.py): Only changed machine-specific configuration lines in the execution version to parameters; the SHA-256 of the original actual execution file remains in the raw evidence index. Uses the machine's existing restricted configuration directory and the same source coordinator package; port 5260/5261/5262 must be free.
- Raw task JSONL, Pi events, generated files, HTTP observation records, and failed preparation stage records are only saved in `artifacts/` ignored by the machine; they are not public download links. Public data does not contain private network addresses, tokens, or private inputs.

Reproduction order: Verify same SHA versions and idle status on both ends → Deployment acceptance → Start observer → Single non-question-group warmup and retain record → Execute fixed 20 question → Official container assertion → Align per-request data. Command entry:

```text
python3 docs/evaluations/main-cd6f7da-20261023/observe-main-run.py --config-dir PRIVATE_MAC_CONFIG --executable SAME_MAIN_COORDINATOR --output NEW_OBSERVER_DIR
python3 scripts/benchmark-pi-humaneval.py --cohort cohort --seed 0 --timeout 300 --context-window 32768 --max-tokens 4096 --endpoint http://127.0.0.1:5260 --endpoint-build cd6f7da401fadf044dac13e87b415d27bb5eadfb --model-revision e06a74e6236a60c8367e1a3214e83d8b61b637b0 --token-file PRIVATE_TOKEN_FILE --output NEW_RUN_DIR
python3 scripts/eval-pi-humaneval-docker.py --results ABSOLUTE_RUN_DIR --dataset PINNED_DATASET --output ABSOLUTE_RESULTS_JSON
```

The preparation phase retained three categories of failures and handling: the first Metal LaunchAgent bootstrap returned 5; after confirming the job was not registered, only partial definitions for this round were cleaned up and reinstalled; the first readiness boundary acceptance failed one item, and after stable readiness, the same out-of-limit request returned 400, re-executing the complete 15 item which passed; the observer initially did not handle chunked body requests, returning 400; after correction, smoke verification was completed before the formal 20 question. They were not mixed into the formal question set nor did they delete formal slow rounds.

The first container call used a relative mount path, and Docker exited with 125, executing no assertions; this error is not the 0/20 task pass rate. After retaining the error file, an absolute path was used instead, and the same unmodified answer completed valid 20 question tests, additionally verifying positive/negative controls actually running. The question answers were not regenerated.

The result added the homepage as the default selected main Run, isolating old configurations according to actual collection standards. The assumption in the original test that "production data always has only two entries" was relaxed: precise historical assertions still lock the original run ID, and new legal records no longer cause false positives in the test; this test adjustment was completed by Copilot CLI `gpt-6.1-sol`, without changing production plotting logic or the inference backend.
