# Experiment 84: bounded CUDA delivery windows

Date: 2026-10-25, America/Los_Angeles. [Issue #84](https://github.com/LittleLittleCloud/Infer.NET/issues/84),
[code PR #86](https://github.com/LittleLittleCloud/Infer.NET/pull/86).
Approved main baseline: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`.
The candidate remains unadopted. Records and implementation have independent PRs.

## Implementation and frozen versions

CUDA completes all20 frontend layers over a bounded32/64/128-token execution
window before emitting ordered32-token wire frames. The existing authenticated
TLS framing, incremental KV, bounded sender and single GPU executor remain.
Metal consumes32 or128 rows. Arithmetic, KV and recurrence stay FP32, with
unchanged affine8/group64 weights and20/20 topology. No Q8, BF16 multiplication,
speculation, cache, concurrency or default switch is included.

The initial implementation exported each KV plane separately for every wire32
frame. Its screens ran `99f7610b640d257601dfcd3c6bab5e51d892d4e2`.
A second iteration exports each completed window's K/V planes once, then slices
immutable wire32 payloads. Its frozen runtime is
`30a3b2c3b2a407975e6cda6fe8677fcbc676e785` on both hosts. The later `53d34a4`
changes evaluation tools only; it is not presented as a measured product binary.
Separate executable/native-library hashes are retained in the public manifests.

The native Windows SM89 checks compare every hidden row, every layer trace and
all40 frontend state tensors against the retained32 path. They cover aligned
and tail lengths1,7,17,32,33,63,64,65,127,128,131 and17 after reset. Both iterations
passed byte equality; the second additionally verifies bulk KV exports against
full-state slices. Managed regression and Native AOT builds passed. These checks
are prerequisites, not full long-context or lifecycle acceptance.

## Workload and measurement

Hardware: M5 Pro20-core GPU/48 GiB and native Windows RTX4090/24564 MiB,
driver610.88. Mac uses AC/High Power. The Mac route remains en0 Wi-Fi and Windows
uses its observed1-Gbit/s Ethernet link. Effective network bandwidth was not
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
Kernel/stage durations overlap and are never added to estimate request duration.
Fixed-request client wall is separate and is not a valid first-output-token TTFT.

## Fixed-request results

Results are inserted from the completed public trial summaries below. All failed
or interrupted control attempts remain in the evidence inventory.

## Resource coverage and 128K preflight

Each fixed request records machine-wide vm_stat swapout deltas separately from
load/warmup. Windows full-device nvidia-smi samples were collected every500 ms;
sampled maxima cannot prove the exact unsampled allocation peak. Worker Metal
allocation is a completed-request observation, not a continuous driver peak or
per-submission residency proof. Model-load swap, process-footprint/memory-pressure
sampling and complete transient-residency evidence were not obtained.

At the observed8192/128 capacity10240, Metal reported37,467,340,800 bytes allocated
and419,430,400 KV bytes. Keeping that observed non-KV allocation and projecting
FP32 KV to capacity135168 (131072+4096) yields42,584,391,680 bytes, already above
the38.5 GiB limit before attention growth and transient resize buffers. This is a
capacity preflight projection under the frozen allocator behavior, not a measured
128K peak or a proof of its minimum allocation. The diagnostic128K requests were
not executed. Q8 experiment33 failed its independent numerical gate and is not
silently substituted. No system memory limit or precision was changed.

## HumanEval and adoption

The fixed5 pilot, any formal20 decision, complete timing coverage and official
assertions are recorded after execution. HumanEval does not replace128K target
acceptance. FP32 arithmetic/KV cannot be relabeled as formal BF16-input/Q8 results.

## Control failures and restoration

The initial controller lost a retained Windows process handle during shutdown;
subsequent recovery also exposed PowerShell's nonzero final status for an already
exited PID. Both were corrected in the experiment controller, retaining original
logs and complete cohorts. The product source was unchanged by these corrections.
A Mac restart then raced the old worker's lock release after its health listener
closed. That startup failed before measurements; waiting for actual process exit
resolved it. No failed startup is included as a speed result.

Original service restoration is recorded after the isolated evaluation completes.
Public summaries omit prompts, generated solutions, tokens, configuration secrets
and private network addresses. Raw data remain local-only in the candidate
checkout's ignored `artifacts/dual-host/issue84-*` directories, identified by
[SHA-256 inventory](evaluations/issue84/raw-evidence.json). These inventory paths
are evidence identifiers, not public download links. Pages sources are reviewable
in this results PR; publication still requires review and an authorized merge.
