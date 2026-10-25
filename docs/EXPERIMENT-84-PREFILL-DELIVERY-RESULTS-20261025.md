# Experiment 84: bounded CUDA delivery windows

Date: 2026-10-25, America/Los_Angeles. [Issue #84](https://github.com/LittleLittleCloud/Infer.NET/issues/84),
[code PR #86](https://github.com/LittleLittleCloud/Infer.NET/pull/86).
Approved main baseline: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`.
The candidate remains unadopted. Records and implementation have independent PRs.

## Implementation and frozen versions

CUDA completes all 20 frontend layers over a bounded 32/64/128-token execution
window before emitting ordered 32-token wire frames. The existing authenticated
TLS framing, incremental KV, bounded sender and single GPU executor remain.
Metal consumes 32 or 128 rows. Arithmetic, KV and recurrence stay FP32, with
unchanged affine8/group64 weights and 20/20 topology. No Q8, BF16 multiplication,
speculation, cache, concurrency or default switch is included.

The initial implementation exported each KV plane separately for every wire32
frame. Its screens ran `99f7610b640d257601dfcd3c6bab5e51d892d4e2`.
A second iteration exports each completed window's K/V planes once, then slices
immutable wire32 payloads. Its frozen runtime is
`30a3b2c3b2a407975e6cda6fe8677fcbc676e785` on both hosts. The later `53d34a4` and `349400b`
change evaluation tools only; neither represents a measured product binary.
Separate executable/native-library hashes are retained in the public manifests.

The native Windows SM89 checks compare every hidden row, every layer trace and
all 40 frontend state tensors against the retained 32 path. They cover aligned
and tail lengths 1,7,17,32,33,63,64,65,127,128,131 and 17 after reset. Both iterations
passed byte equality; the second additionally verifies bulk KV exports against
full-state slices. Managed regression and Native AOT builds passed. These checks
are prerequisites, not full long-context or lifecycle acceptance.

## Workload and measurement

Hardware: M5 Pro 20-core GPU/48 GiB and native Windows RTX 4090/24564 MiB,
driver 610.88. Mac uses AC/High Power. The Mac route remains en0 Wi-Fi and Windows
uses its observed 1-Gbit/s Ethernet link. Effective network bandwidth was not
measured independently; exposed wait cannot be attributed to network alone.

Model: `mlx-community/Qwen3.6-35B-A3B-8bit` at
`e06a74e6236a60c8367e1a3214e83d8b61b637b0`. No prefix cache or speculation.
Only one model instance per GPU was used during valid timing cohorts. A fixed
request uses one complete warmup plus three formal repetitions, with natural EOS
ignored to obtain the specified output length. The pinned prompt hashes, actual
runtime build, driver hashes, individual slow rounds and warmups are preserved in
[fixed-requests.json](evaluations/issue84/fixed-requests.json).

Complete prefill is the real worker duration including CUDA delivery, Metal
suffix, state wait/import and handover. Decode counts N−1 first-to-last output
intervals. Aggregate rates divide summed tokens/intervals by summed phase time.
The agent field `cudaComputeMs` is host stopwatch time around native execution,
synchronous hidden/KV readback and frame preparation; it is not a pure CUDA-event
kernel duration. GPU-kernel versus D2H separation was not obtained for these
service cohorts. Stage durations overlap and are never added to estimate request duration.
Fixed-request client wall is separate and is not a valid first-output-token TTFT.

## Fixed-request results

| CUDA / Metal | Prefill median (s) | Aggregate prefill (tok/s) | Aggregate decode (tok/s) |
| --- | ---: | ---: | ---: |
| 32 / 32 | 16.080 | 489.01 | 60.16 |
| 64 / 32 | 16.088 | 509.15 | 59.73 |
| 128 / 32 | 16.236 | 477.20 | 60.24 |
| 32 / 128 | 11.644 | 699.16 | 59.99 |
| 64 / 128 | 11.745 | 697.14 | 59.39 |
| 128 / 128 | 11.600 | 700.32 | 59.44 |

The frozen selection is 128/128. Its 0.38% lower complete-prefill median versus
the retained 32/128 path is below the 5% gate and is not evidence of a gain over
main. CUDA compute medians fall 10.079→9.598 s while the Metal suffix remains
10.317→10.318 s; hidden wait rises 0.712→0.804 s. These overlapping stage observations are consistent with the lack of complete
prefill benefit; they do not isolate network, CPU scheduling or GPU residency costs.

All 24 screen requests, including warmups, produce the same 128 output token IDs
(SHA-256 `cb1c4d9d899d2cfc5ec9c2578cb2bd89425a61bf73fd931774658c1f312feb6b`).
The initial wire32-readback iteration retains its three complete screens at
11.433/11.500/11.603 s for CUDA 32/64/128 with Metal 128. The bulk-readback
iteration is a distinct run; variations between phases are not attributed solely
to code. All failed control attempts and slow formal samples remain in evidence.

## Matched current-main controls

Both hosts were freshly built and verified at main
`4f51f241e87999c8cc3f79d733cd95ade9c711cd`; the frozen candidate remains
`30a3b2c3b2a407975e6cda6fe8677fcbc676e785`. Separate main and candidate binary
manifests are published beside the trial summaries. All shapes use the same
prompt hash, FP32 arithmetic/KV, packed CUDA prefix, incremental transfer, bounded
send overlap, Metal 128, no cache/speculation, one warmup and three formal requests.

| Input / output | Main prefill median (s) | Candidate (s) | Candidate latency change | Main decode (tok/s) | Candidate decode (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192/128 | 11.685 | 11.600 | -0.733% | 60.081 | 59.442 |
| 32768/128 | 53.109 | 54.090 | +1.847% | 44.437 | 44.659 |
| 32768/4096 | 53.296 | 53.957 | +1.239% | 45.320 | 45.068 |

Negative latency change is faster. Decode columns use summed N−1 intervals over
summed decode time, not average request rates. For every shape, all eight output
token sequences (four per arm) agree exactly. The larger six-way screen and
initial iteration also retain their complete slow trials. Public
[current-main comparisons](evaluations/issue84/current-main-comparison.json)
include individual run IDs and input/output hashes. The full
[trial summaries](evaluations/issue84/fixed-requests.json) include raw warmups,
formal means, medians and slowest times; no p95 is inferred from three samples.

Candidate and main ran as serial blocks, rather than randomized/interleaved arms;
network effective bandwidth was not separately measured. The sub-percent change
at 8192 is not claimed statistically significant. None of the three fixed shapes
improves complete prefill median by 5%. Prefill/decode regressions remain below 5%,
but that does not waive the positive-gain requirement. The registered fixed-gain
prerequisite therefore fails; formal HumanEval 20 is not run, and this prototype
is not qualified for adoption. The default 32 path and adopted main remain unchanged.

## Resource coverage and 128K preflight

Each fixed request records machine-wide vm_stat swapout deltas separately from
load/warmup. Windows full-device nvidia-smi samples were requested at 500 ms intervals;
sampled maxima cannot prove the exact unsampled allocation peak. Worker Metal
allocation is a completed-request observation, not a continuous driver peak or
per-submission residency proof. Across the retained fixed cohorts, the maximum completed-request Metal allocation
was 39,327,547,392 bytes (36.627 GiB); maximum formal-request new system swap-out
was 141,492,224 bytes (134.938 MiB), and maximum warmup delta was 17,367,040 bytes.
The 7936 full-device Windows samples peaked at 20539 MiB (20.058 GiB). These
observations are below the respective limits but cannot establish missing peaks.
[Resource coverage](evaluations/issue84/resource-coverage.json) distinguishes
samples from acceptance. Model-load swap, process-footprint/memory-pressure
sampling and complete transient-residency evidence were not obtained. The full
resource gate remains incomplete.

At the observed 8192/128 capacity 10240, Metal reported 37,467,340,800 bytes allocated
and 419,430,400 KV bytes. Keeping that observed non-KV allocation and projecting
FP32 KV to capacity 135168 (131072+4096) yields 42,584,391,680 bytes, already above
the 38.5 GiB limit before attention growth and transient resize buffers. This is a
capacity preflight projection under the frozen allocator behavior, not a measured
128K peak or a proof of its minimum allocation. The diagnostic 128K requests were
not executed. Q8 experiment 33 failed its independent numerical gate and is not
silently substituted. No system memory limit or precision was changed.

## HumanEval and adoption

The fixed five tasks (HumanEval/156, /94, /72, /80, /149; seed 0) all completed
and passed their official assertions. Pi 1.4.0 used the pinned official dataset
SHA-256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`,
context 32768, output cap 4096 and timeout 300 s. No retries or failures were dropped.
The offline Docker evaluator used its pinned Python image and a failing negative
control; all five solutions passed. The evaluator was extended to support the
approved fixed 5 size without changing assertions.

| Pilot measurement | Result |
| --- | ---: |
| Pi task completions / official assertion passes | 5/5 / 5/5 |
| Model requests / terminal worker clocks / valid-output TTFT samples | 21 / 21 / 21 |
| Input / output tokens | 49966 / 4768 |
| Summed complete worker prefill | 90.0370275 s |
| Summed worker decode / N−1 intervals | 71.438522 s / 4747 |
| Aggregate worker prefill / decode | 554.949 / 66.449 tok/s |
| Median first-valid-output client TTFT | 7476.933 ms |
| Total Pi task wall time | 163.381726334 s |
| Timeouts / application errors | 0 / 0 |

`metrics_complete=true`: every Pi request reconciles its input/output counts with
raw worker `ServiceEvent.result.timings`. The observer normalized these genuine
clocks into the summarizer's terminal-metric schema; this is explicitly observer
evidence, not a claimed native INFERENCE_METRIC log from this binary. No dashboard
snapshot, Pi wall or per-request average throughput substitutes for worker time.
The public [pilot summary](evaluations/issue84/pilot5.json) includes every request
and task. Actual coordinator, Mac and Windows runtime source is the same frozen
30a3b2c SHA; the harness source is 53d34a4, with individual tool hashes preserved.

TTFT starts when the loopback observer forwards the request to the real coordinator
and ends on the first nonempty text/tool-call output; role, usage and keepalive
events are ignored. Pi-to-observer dispatch is excluded. Two loopback streaming
hops are disclosed as a separate configuration. Tool-call buffering can put this
TTFT after most model decoding; internal first-token readiness is not substituted.
The pilot follows the long fixed shapes in the same resident worker and retains
capacity 36864. It is an iteration/health result, not a main comparison or an
adoption decision. Formal 20 was not run because the matched fixed-prefill gain prerequisite failed.
HumanEval does not replace 128K target acceptance. FP32 arithmetic/KV cannot be
relabeled as formal BF16-input/Q8 results.

## Functional checks

The candidate passed all 15 existing deployment checks on the measured runtime:
authentication/limits, deterministic resets, streaming/nonstream equality,
cancellation after first output and the next correct request, natural EOS,
bounded queue backpressure and unchanged Mac model PID. Public
[lifecycle summary](evaluations/issue84/candidate-lifecycle-summary.json) preserves
the actual checks and observed queue statuses. After the final timing cohort, a fresh instance of the same frozen binary also
passed two [additional checks](evaluations/issue84/extra-lifecycle-summary.json):
an authenticated 8192-token request was disconnected after Busy, produced no
completed result, and recovered to the same next-request token IDs without Mac
reload; stopping the owned agent revoked readiness and rejected a request with 503,
then restarting it restored healthy registry status and exact token IDs with the
same Mac PID. These are functional observations, not throughput trials.

## Control failures and restoration

The initial controller lost a retained Windows process handle during shutdown;
subsequent recovery also exposed PowerShell's nonzero final status for an already
exited PID. Both were corrected in the experiment controller, retaining original
logs and complete cohorts. The product source was unchanged by these corrections.
A Mac restart then raced the old worker's lock release after its health listener
closed. That startup failed before measurements; waiting for actual process exit
resolved it. No failed startup is included as a speed result.

After evaluation, the original deployment
`cd6f7da401fadf044dac13e87b415d27bb5eadfb` was restored: Mac login LaunchAgent,
original Windows foreground agent held by SSH (no SCM claim), and unchanged
original coordinator/TUI. Original binaries, private configurations and plist
hashes match the pre-experiment record. Two real split requests returned identical
32-token outputs; worker readiness and fresh accepting registry telemetry are
healthy. All six temporary listening ports are closed and wired limit 0 is unchanged.
The first coordinator request returned 503 before final acceptance; its body was
not captured. That failure is retained, followed by the stricter fresh-PID/
accepting-telemetry check and the two successful requests. It is not erased as a
startup retry. [Restoration evidence](evaluations/issue84/restoration.json) records
both observations. Neither P0 candidate was adopted.
The Windows evidence collector retained 28 log/exit files (including the initial
99f-only 2068-line extract from the original tee log). Its first collection completed
but a final status-print variable mistake returned a Python error; retry then failed while restoration logs were active; its remote stderr was not
captured. Excluding active restoration logs allowed collection to complete,
consistent with a file-sharing conflict but not proving the exact remote cause.
The observations and missing error detail are retained. This affected
metadata collection after all timing, not source code or benchmark samples.

Public summaries omit prompts, generated solutions, authentication tokens, configuration secrets
and private network addresses. Raw data remain local-only in the candidate
checkout's ignored `artifacts/dual-host/issue84-*` directories, identified by
[SHA-256 inventory](evaluations/issue84/raw-evidence.json). These inventory paths
are evidence identifiers, not public download links. Pages sources are reviewable
in this results PR; publication still requires review and an authorized merge.
