# Experiment #58: M5 Pro Fully Local Pi / HumanEval 20

Measurement date: 2026-10-24 (America/Los_Angeles). [Issue #58](https://github.com/LittleLittleCloud/Infer.NET/issues/58) · [Code PR #59](https://github.com/LittleLittleCloud/Infer.NET/pull/59). Results PR: https://github.com/LittleLittleCloud/Infer.NET/pull/60。

The user first specified the latest main, fixed 20 problems, and disabled 4090 for prefill, then explicitly approved adding fully local routing and testing the candidate SHA. The chat service on the original main only allowed ≤128-token local prefill, and standard workers rejected structured chat. The candidate opened the configurable `short-prefill-tokens` limit to the model window, explicitly setting it to 262144; default 128 and 0 force splitting, while model precision and kernels remain unchanged. This round did not perform a same-configuration A/B test against the original main, so no code speedup percentage is provided.

## Versions and Configuration

- Fixed latest main baseline: `eaaf81f630614b4d6416bc62776c13d36331fa57`.
- Actual running coordinator / Mac serve / Metal library: candidate `87c2b7dcaf4a418dcb18707e1913562717b43cdc`; Windows is not involved. The code PR later `19162c7` only updated integration tests and did not masquerade as a tested running version.
- M5 Pro / 48 GB, AC / High Power; independent Native AOT package and port. The original Mac worker was deactivated via LaunchAgent after becoming idle, and the original package was restored after testing ended. No second model worker contended for the GPU.
- `mlx-community/Qwen3.6-35B-A3B-8bit`, revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine8/group64, FP32 KV/state, fused FP32 decode, Metal residency on. Fixed weight hashes were verified file-by-file at startup.
- Retained managed prefill chunk 32 and scratch 128 for the deployment path; this round only changed routing, not switching to the independent high-throughput BF16/2048 experimental configuration.
- Fixed 20 problems, seed 0, Pi 1.4.0, new working directory/session; context 32768, max output per request 4096, per-problem timeout 300 seconds, single request serial. State was cleared for each request, with cacheRead/cacheWrite both at 0.
- Official data revision `b9453e8f1865df0a0057a1383767caa400acfbcf`, compressed file SHA-256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`. Full problem IDs are available in the public configuration and per-problem table.
- After startup warmup, two additional 518-token structured chats were performed: both were metal-local, with consistent output token IDs; these were not included in the cohort.

## Results

Formal problem group time: 10:51:21–11:11:24 (America/Los_Angeles). Run ID: `issue58-m5pro-local-human20-20261024`.

|Metric|Measured|
|---|---:|
|Pi completion / Official assertion pass|20/20 · 20/20|
|Timeout / Pi error|0 / 0|
|Model requests / Worker metric coverage|107 / 107 (100%)|
|metal-local / CUDA or cross-machine transfer requests|107 / 0|
|Input / Output tokens|243663 / 19104|
|Prefill total time / Throughput|908.496 s / **268.205 token/s**|
|Decode interval / Total time / Throughput|18997 / 285.955 s / **66.434 token/s**|
|Client TTFT median / Coverage|10914.875 ms / 107/107|
|Total Pi task duration|1202.992 s (20.05 minutes)|

Mac executable SHA-256：`b4a15505fd2c86f0cd44185cd982eda666bfc5fff9a64e6f7cda494fbd2e9df8`；Metal library SHA-256：`1739ff5dd22a4b158f97d84d39268478a75d5683f4a9ad1a3cd0d1a8f79d4aef`。

Prefill uses Σworker input tokens / Σworker prefill seconds; decode uses Σ(N−1) / Σworker decode seconds. Per-request raw `ServiceEvent.result.timings` was saved from real SSE and reconciled item-by-item with the number of requests, input/output tokens per Pi problem; throughput was not derived from Dashboard or Pi wall time. There was no CUDA computation, transfer, or state import.

Client TTFT is timed by the observation agent from before forwarding to actually receiving the first valid text/tool output, ignoring role, heartbeat, and usage events. This configuration adds two loopback observation forwards; dispatch from Pi to the observation agent is not within the TTFT interval, so this number is interpreted separately from other timing boundaries.

## Correctness, Resources, and Recovery

The official assertion uses fixed image `python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd`, running each problem in an isolated container with network disabled, read-only, unprivileged, and CPU/memory limits. 20/20 assertion passed; the erroneous implementation failed as a negative control. The first call passed a relative path by the executor; all 20 containers returned 125 due to invalid bind mount paths, so no problems were executed. `functional-mount-error.json/.log` was retained and not counted as a problem assertion failure. After re-checking with the same unmodified solution.py and absolute paths, all passed; no problem answers were regenerated or selected.

The original Mac LaunchAgent has been restored, PID changed to 16997, build `0.3.0+cd6f7da401fadf044dac13e87b415d27bb5eadfb`, status Ready. Candidate models, coordinator, observer, and this round's Pi processes have all exited; the candidate package was not replaced with a persistent service.

131 .NET unit tests and 44 Native AOT HTTP/process integration checks passed. Integration coverage threshold 0/128/129/262144 was effectively configured, and 262145 was rejected. Model configuration/execution paths, repeated long chat state resets, and package hash verification passed.

Resource sampling over the entire round of 10 seconds consistently showed AC; additionally, a boundary was sampled approximately 0.117 seconds after the problem set ended, while the candidate model was still alive and the original service had not yet reloaded. Metal allocation peak was **34.730 GiB**. The pre- and post-problem-set sampling boundaries were `2026-10-24T17:51:18.457138+00:00` and `2026-10-24T18:11:24.974177+00:00`, with the whole machine experiencing a new swap-out of **2.013 GiB**; loading/warm-up accounted for another **8.688 GiB**. This is not a "no swap" result, nor can it be marked as resource acceptance passed. Swap-out is a whole-machine count, including background processes, and cannot be attributed to a single process; existing swap usage does not count as a new value for this round.

Resource values come from the problem-set interval bracketed by sampling; precise per-request swap-out or process footprint peaks were not obtained. Actual RSS is not used as a footprint substitute. Memory pressure and significant paging are the limiting factors for performance in this round; complete raw sampling and all formal requests are retained, excluding no slow samples.

This is a single fixed cohort measurement of application load, not representing the 128K formal target acceptance, nor proving the candidate is superior to the dual-machine path. Historical points with different topologies, precisions, chunking, or TTFT boundaries are preserved in different configuration series; no fabricated comparable A/B baselines were created. The user explicitly approved the merge of two PRs at 2026-10-24. Code PR #59 has been merged into main via merge commit `9d525237eac265b1fe5c4ccf2649a4b6a536d682`, preserving the ancestry of the tested `87c2b7dcaf4a418dcb18707e1913562717b43cdc`. This measured point is marked as adopted in the chart and mapped to the merged version; the merged artifact was not re-measured, no fictional merged SHA measurement points were added, and it was not automatically set as the re-measured main baseline for this series. Result PR #60 was merged according to this authorization; the Issue will be closed after results are merged and Pages verification is complete.

## Reproduction and Evidence

```sh
# Build, package, and verify the measured code PR SHA, then set in a separate private configuration:
# short-prefill-tokens=262144, kv=fp32, decode-attention=fused-fp32,
# metal-scratch-tokens=128, data-listen=loopback-only。
# In the private configuration, coordinator HTTP=5290 and worker HTTP=5291 use the same token-file.
# Start coordinator / split-serve, wait for Ready, then run the observer (5280/5281/5282):
python3 docs/evaluations/issue58/observe.py --config-dir PRIVATE_MAC_CONFIG \
  --executable VERIFIED_PACKAGE/deepinfer --output NEW_RUN/cohort-observer
# Run the fixed cohort in another terminal:
python3 scripts/benchmark-pi-humaneval.py --output NEW_RUN/cohort-20 \
  --token-file PRIVATE_TOKEN_FILE --endpoint http://127.0.0.1:5280 \
  --cohort cohort --seed 0 --timeout 300 --context-window 32768 --max-tokens 4096 \
  --endpoint-build 87c2b7dcaf4a418dcb18707e1913562717b43cdc \
  --model-revision e06a74e6236a60c8367e1a3214e83d8b61b637b0
python3 scripts/eval-pi-humaneval-docker.py --results NEW_RUN/cohort-20 \
  --dataset PINNED_HUMANEVAL_GZ --output NEW_RUN/functional.json
python3 scripts/import-issue58-local-humaneval.py --run-dir NEW_RUN --results-pr https://github.com/LittleLittleCloud/Infer.NET/pull/60
```

The parameterized observation script is at [observe.py](evaluations/issue58/observe.py). Public values are at [performance.json](evaluations/issue58/performance.json), evidence hashes at [evidence.json](evaluations/issue58/evidence.json). Per-problem and per-request data are saved in [site/trends.json](../site/trends.json). Raw logs, Pi sessions, and generated code are only in the local ignore directory `artifacts/humaneval20-m5pro-20261024-preflight/`, not public download links; the public summary does not contain credentials, private network addresses, full prompts, or output bodies.

Stable measurement point: `https://littlelittlecloud.github.io/Infer.NET/#experiment=humaneval20-m5pro-20261024&run=issue58-m5pro-local-human20-20261024`. This point is published by the existing Pages workflow after the result PR is merged; it is only marked as live after the workflow and actual page are successfully verified.
