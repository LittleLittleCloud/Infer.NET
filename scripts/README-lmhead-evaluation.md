# Experiment #103 evaluation harness

**Execution attempted; repaired-host rerun pending parent review.** Retain the
local-only `battery-assisted-20261027-*` directories unchanged. The
`battery-assisted-20261027-ready-fixed` attempt reached pilot-off, then stopped
on a Pi/worker count mismatch; it did not complete main/off/on fixed20.
This implementation stage ran only CPU fake-worker/Pi tests and compilation.
This controller never builds packages, changes power/sysctl, edits LaunchAgents,
or controls the original coordinator/Windows. It is not 128K acceptance.

The earlier product correction in `SplitService.Profile` preserves
the managed `DEEPINFER_METAL_LMHEAD` selector while clearing other experimental
environment settings. Without this, `split-serve` would erase candidate-on.
The API host now also handles invalid model output explicitly and preserves
known worker usage on failures (see protocol semantics below). The parent must
freeze/repackage the repaired source before running; historical packages do not
include this repair. No kernel, quantization, sampling or tool-prompt change is
included. The source-pinned external Pi driver/evaluator remain unchanged.

## Preparation and commands

Run from the repository root. Replace uppercase placeholders with verified
absolute paths/full hashes after the parent freezes the code. Every invocation
needs a **new, nonexisting** output directory under this checkout's ignored
`artifacts/`. Files use restricted permissions; configurations, request artifacts,
Pi output and recovery commands remain local-only. Never publish raw directories.

```sh
# Safe: CPU only, owned Python children/loopback fakes; no real services, Docker, model or GPU.
python3 scripts/test-lmhead-evaluation.py -v
dotnet test tests/DeepInfer.Tests/DeepInfer.Tests.csproj --no-restore \
  --filter 'FullyQualifiedName~Chat' -v quiet
dotnet build src/DeepInfer.Host/DeepInfer.Host.csproj -c Release --no-restore -v quiet
python3 scripts/test-chat-termination.py --pi pi -v
python3 scripts/chat-integration.py
python3 -m py_compile scripts/issue103_common.py scripts/*lmhead*.py scripts/test-chat-termination.py
dotnet build DeepInfer.slnx --no-restore

# Read-only preflight; never pauses the worker or submits model/GPU work.
# Executes package "version", Pi "--version", Docker image inspect, health/idle
# reads, file hashing and read-only resource/power queries.
python3 scripts/run-lmhead-evaluation.py \
  --phase preflight \
  --candidate-package CANDIDATE_RELEASE/deepinfer \
  --candidate-sha CANDIDATE_FULL_SHA \
  --candidate-source FROZEN_CANDIDATE_CHECKOUT \
  --common-coordinator-package REPAIRED_RELEASE/deepinfer \
  --common-coordinator-sha REPAIRED_HOST_FULL_SHA \
  --common-coordinator-source FROZEN_REPAIRED_HOST_CHECKOUT \
  --output artifacts/dual-host/issue103-lmhead-affine8/preflight-UNIQUE
```

For this rerun, use the three explicit `--common-coordinator-*` arguments.
They are optional as a group: without them, the original same-arm coordinator
selection remains unchanged. With them, **both** controller-owned loopback
coordinators in **every** main/off/on arm use the exact same verified repaired
executable. The common host can be the new frozen candidate package/source;
its full SHA is still recorded separately as coordinator provenance.
The main **worker** remains immutable
`4f51f241e87999c8cc3f79d733cd95ade9c711cd`; off/on workers use the frozen
candidate package, never the common host argument as an implicit worker override.
The deployed original coordinator is not changed.

Preflight verifies the common source HEAD and a clean tracked/untracked status,
manifest, executable/native SHA-256 and executable version. Source cleanliness
and package hashes are revalidated before bootout and before/after each arm's
startup. Runtime evidence records both owned coordinator process identities,
roles, actual executable paths and package provenance separately from the worker.
Preflight, plan, gate and pilot identities bind these fields; old evidence or
evidence from a different host cannot authorize a new phase. Summaries reject
mixed runtime coordinator provenance.

This comparison is **unchanged main inference under a common repaired API
host**, not the original main coordinator. Candidate off/on isolate the LM-head
layout; main-to-on also includes candidate worker plumbing. All other fixed
settings remain unchanged.

Default external paths are the user-provided main package, independent #84
harness checkout, #98 fixed fixtures, private Mac config and parent original
preflight record. `--help` lists overrides. Preflight pins:

- Actual main `4f51f241e87999c8cc3f79d733cd95ade9c711cd`, not candidate-off.
- Independent harness checkout `349400b92b9de5efb88d73ce8cf5931f94f9e7b7`;
  driver, standard summarizer and official evaluator file hashes in
  `issue103_common.py`. No inference code is imported from it.
- Pi 1.4.0, seed-0 pilot/fixed20 IDs, dataset compressed hash/revision and
  offline Docker image from the approved plan. No model tasks in preflight.
- All 11 pinned model/config/tokenizer hashes, immutable package manifest,
  executable/native hashes, prompt hashes and exact count/token-ID manifests.
  Actual runtime input counts are checked on every fixed request.

The parent supplies a fresh, private JSON file after physically connecting
the selected policy's AC adapter and re-verifying the settings UI, for example:

```json
{
  "utc": "ACTUAL_CURRENT_UTC_ISO8601",
  "verifiedBy": "parent",
  "source": "Actual System Settings Battery/Energy Mode verification",
  "energyModeControls": ["High Power", "High Power"]
}
```

This is a schema example, **not verification evidence**. Evidence must be at
most 30 minutes old at initial admission and immediately before bootout.
The default `--power-policy stable-140w` still requires 140W AC and no battery
decline greater than one percentage point from admission.
The user renewed authorization after the API repair for an exploratory
full-battery 35W experiment, to be run by the parent only after review:
select `--power-policy battery-assisted-35w` for initial AC35W and SOC >=99%,
unchanged adapter identity and AC35W throughout, SOC >=80%, and at most a
20-percentage-point drop from admission. This is exploratory, **not stable-140W
acceptance**. No automatic retries are authorized. Both policies retain the fresh
parent High Power UI gate; the controller does not change OS settings.
Explicit `pmset -g therm` CPU speed/scheduler limits below 100 or thermal
warning values stop execution. Missing limit lines and the normal
`No thermal warning level has been recorded` are not warnings. Raw diagnostics
are retained. Policy and thresholds are recorded in preflight, plan and bound
evidence identity, and carried into summaries.

Compile the native gate driver from the frozen candidate source without running
it (the existing `check-metal-lmhead.sh --compile NEW_DIRECTORY` does this).
Pin its executable SHA-256 and source SHA. Then, **only after exclusive GPU
scheduling and paused clients**, the parent may run:

```sh
python3 scripts/run-lmhead-evaluation.py \
  --phase all --execute --exclusive-gpu --clients-paused \
  --candidate-package CANDIDATE_RELEASE/deepinfer \
  --candidate-sha CANDIDATE_FULL_SHA \
  --candidate-source FROZEN_CANDIDATE_CHECKOUT \
  --common-coordinator-package REPAIRED_RELEASE/deepinfer \
  --common-coordinator-sha REPAIRED_HOST_FULL_SHA \
  --common-coordinator-source FROZEN_REPAIRED_HOST_CHECKOUT \
  --native-driver FROZEN_NATIVE_DRIVER \
  --native-driver-sha256 NATIVE_DRIVER_SHA256 \
  --native-driver-source CANDIDATE_FULL_SHA \
  --high-power-evidence PRIVATE_HIGH_POWER_JSON \
  --output artifacts/dual-host/issue103-lmhead-affine8/evaluation-UNIQUE

python3 scripts/summarize-lmhead-evaluation.py \
  --run artifacts/dual-host/issue103-lmhead-affine8/evaluation-UNIQUE \
  --output artifacts/dual-host/issue103-lmhead-affine8/evaluation-UNIQUE/summary.json
```

Separate phases use the same execution arguments with `--phase gates|fixed|pilot|cohort`.
`fixed` and `pilot` additionally require `--gates-evidence PRIOR_GATES_RUN/gates.json`.
`cohort` requires both that argument and
`--pilot-evidence PRIOR_PILOT_RUN/completed.json`. Evidence must match both
immutable packages, source, harness, native driver and input hashes. Execute
the scripts from the clean frozen candidate checkout itself.
Separate runs are summarized separately; do not combine
incomplete rounds as if they were one formal cohort.

### Parent freeze/build handoff

Review the diff, then freeze it on the feature branch using the parent's
normal commit/PR process; no commit/push is performed in this implementation
stage. From that clean full SHA, publish Native AOT into a **new** staging
directory using `dotnet publish src/DeepInfer.Host -c Release -r osx-arm64 -o NEW_STAGING`,
then run `scripts/deploy/package.py --publish-dir NEW_STAGING --rid osx-arm64 --output NEW_RELEASE`
and `scripts/deploy/verify-package.py NEW_RELEASE` with Python. Do not overwrite
any historical package or deployed service. Run
`python3 scripts/test-chat-termination.py --executable NEW_RELEASE/deepinfer --pi pi -v`
against the packaged AOT host (still CPU fake-worker only). Rebuild the native gate driver
with `scripts/check-metal-lmhead.sh --compile NEW_DRIVER_DIRECTORY`; record
its full source SHA and executable SHA-256. These build instructions are not
GPU execution or deployment authorization.

Use the new candidate package as the common host in all arms (or explicitly
pin a separately frozen repaired host). Start a **new** evidence run, repeat
gates and pilot5 before fixed20, and do not reuse historical gates/pilot identity.
After review, the parent selects `--power-policy battery-assisted-35w` and
supplies fresh High Power UI evidence; initial SOC >=99%, SOC floor 80%,
maximum 20-point drop, unchanged AC35W/adapter, thermal, resource and integrity
stops are unchanged. Slow or unsuccessful model tasks alone do not cancel20;
infrastructure, coverage/count or correctness failures still stop explicitly.

## Compatible API termination and accounting

In the retained pilot-off final HumanEval/149 request, the worker completed
with prompt3645/output151/EOS. `ChatTools.ParseOutput` rejected model tool text;
`CompatibleChat.Writer.End` let `InvalidDataException` escape because it is
not an `IOException`. Pi consequently reported a stream without a finish reason
and usage0/0. Those zeros were missing accounting, not zero model work.

**Evidence limitation:** the observer deliberately retained counts, timing,
finish reason and token-ID hashes, not output text or raw token IDs; tool-enabled
streaming buffered the text before rejection. The retained Pi/worker/controller
artifacts contain no recoverable rejected tool text. Its precise malformed
syntax cannot be established from this evidence. No parser extension or claimed
syntax repair is justified. The sanitized synthetic regression uses a declared
`read` call with a complete parameter/function but a missing `</tool_call>`;
this reproduces the exception path, not the historical private contents.
Additional parser negatives reject incomplete wrappers, undeclared/duplicate
arguments, trailing text and inter-call garbage.

- Success uses only worker `eos` -> `stop`/`end_turn`/`completed`, or `length` ->
  `length`/`max_tokens`/`incomplete`; valid structured tools use `tool_calls`.
  Unknown finish reasons, missing token IDs and contradictory stream/terminal
  counts fail explicitly rather than becoming length completions.
- A terminal worker result supplies exact usage even when tool conversion fails.
  Streaming Chat Completions sends the requested usage-only chunk **before**
  the error event, because OpenAI clients throw on that event. The error also
  includes usage. There is no tool-call delta, success finish reason or `[DONE]`
  on failure. Nonstream conversion failures return HTTP 502 JSON error plus
  genuine usage, not a successful assistant message.
  The `invalid_model_output` code identifies a rejected tool call; worker
  metadata/accounting errors use `invalid_worker_response`, and transport
  failures use `worker_request_failed`. The observer retains only these
  sanitized codes, not rejected arguments. Accounted model failures remain
  failures in the task denominator; worker/infrastructure errors still stop.
- Without a terminal result, total generated usage is unknown. Errors omit
  exact usage and include `observed_usage` with known prompt count (or null),
  `output_tokens_at_least` from received token events and `complete:false`.
  These are lower-bound observations, not billable/exact totals. No tokenizer
  estimation, zero-filling or count reconciliation by fabrication is used.
- Messages streams terminate failures with `error`, never `message_stop`;
  Responses streams use `response.failed` with failed status and known usage
  or explicit partial observations, never `response.completed`. Input errors
  remain 4xx; malformed upstream/model output is 502 before headers or an explicit
  terminal error after headers. Client cancellation cannot guarantee delivery
  of a final event and does not synthesize usage or completion.

`scripts/test-chat-termination.py --pi pi -v` exercises the installed Pi 1.4.0
against the real managed loopback coordinator and a CPU fake worker with the
same provider compatibility settings as the frozen driver. It checks success,
length, empty EOS, a valid tool/result round-trip, malformed tool failure with
genuine 3645/151 usage and **no tool execution/retry**, upstream faults and
truncation. Private prompts, rejected tool arguments and credentials are not
published. Live-model behavior and Native AOT runtime acceptance still require
parent verification; CPU tests do not establish benchmark correctness or gains.

CPU verification of this working-tree repair (not a frozen package): the
focused managed Chat suite passed 32 tests; the evaluation harness passed all
49 existing plus 10 new CPU tests; the new protocol suite passed 6 tests,
including 7 Pi scenarios; the existing chat integration passed 37 checks.
Release host and solution builds completed without warnings/errors, and the
Python compile checks passed. The commands above reproduce these checks.

## Frozen protocol and safety

Main leaves LMHEAD absent; candidate-off selects `off`; candidate-on selects
`tile4`. All other inference settings are identical: affine8/group64, FP32
KV/operands/state, managed chunk32/native scratch128, matrix0/sequential Delta,
explicit fused-FP32 decode attention, residency on and local threshold262144.
Inherited experimental/native-loader settings are removed, and effective
native profile/selector/bytes are mandatory. Main's absence of candidate buffers
is established by its verified main package; it cannot emit a new candidate
metadata API. Candidate-off must log zero extra bytes. Native on/off mismatch
is an error, never a fallback.

Default stable-policy order: gates, fixed, pilot, cohort. Battery-assisted
`--phase all` prioritizes HumanEval: gates, pilot, cohort, fixed.
The selected phase order is recorded in the plan. Fixed uses five shapes
main/on with AB/BA shape-wise counterbalancing; pilot uses main/off/on;
fixed20 uses on/off/main. Each fixed arm/shape has one excluded
warmup and three retained formal trials. Only fixed-length requests ignore EOS;
Pi handles EOS naturally and retains its 300-second per-task timeout. Fixed
requests and both observation hops allow up to 3600 seconds; the worker's
supported deadline is 1800 seconds. No automatic benchmark retries or
speed-based pilot stopping. Assertion failures/timeouts remain in the task
denominator; slow/unsuccessful tasks alone do not cancel the requested20.
Correctness, resource or infrastructure failures still stop and retain evidence.

Gates run the frozen native `--gpu` driver, then main/off/on short/8K exact
generated sequences, repeated reset, natural EOS and real first-token client
cancellation followed by exact recovery. A cancellation is intentionally not
counted as a completed timed request. The bounded microbenchmark remains the
separate existing `check-metal-lmhead.sh --bench` command under exclusive
scheduling; it is not silently mixed with clean application throughput.

The observer adapts the pinned #98 collector, removing speculation fields and
making both coordinators controller-owned. All arms use two identical loopback
hops. Before any request, observer readiness waits for the cloned registration to be
healthy, explicitly accepting requests, advertising the requested chat-v1 model
and an inference-capable Metal device, with the verified arm's worker ID/PID/build.
`ready-registration.json` retains only sanitized identity/eligibility fields;
readiness never probes or retries inference.
Standard summary inputs are explicitly labeled
`ServiceEvent.result.timings`, converted to the pinned `INFERENCE_METRIC` schema;
they are **not fabricated native worker logs**. Worker decode sums N-1 intervals,
not per-request rate averages. Pi counts, worker counts, delivered raw fixed
tokens, request IDs, durations, task time windows and cumulative Pi usage are
reconciled. Missing/error durations and incomplete coverage yield null aggregate
rates. Observer first-valid-text/tool TTFT excludes Pi-to-observer dispatch;
raw-token TTFT, Pi wall and worker clocks are separately labeled.

Official evaluation uses the pinned external evaluator. Before each cohort/arm,
both a successful Python execution and an actual assertion-negative fixture must
run in the pinned container. Docker exit125/126/127 cannot count as a model
failure. All official rows, solutions, failures and raw logs are retained.

Before bootout: verify immutable packages/model/fixtures, exact original
PID/path/command/build and LaunchAgent identity, parent-saved plist/config/binary/
native hashes, three idle/zero-queue samples, free experiment ports, frozen
source, power and UI evidence. Save `recovery-plan.json` **before** calling
bootout on only `com.infernet.metal`. Wait for that identity to exit and for
three resource-quiescent samples, using the reference admission thresholds.
Original configs/definition are never modified. Experimental data listeners are
loopback-only; Windows cannot attach to them.

Every exception/SIGINT/SIGTERM/SIGHUP passes through cleanup/restoration.
Only recorded child identities are signaled, with bounded graceful waits and
specific-PID escalation. Pi gets SIGINT so its established harness cleans up its
own process tree. Startup records three matching full process identities,
sampling 50 ms apart with a two-second stabilization deadline to accommodate
launcher re-exec. Sampling never polls/reaps the child, preventing PID reuse;
subsequent cleanup still requires exact PID, start time, executable and full
arguments. CPU regressions exercise real Python spawn/cleanup and two-hop fake
HTTP discovery/registration, GET forwarding and SSE timing/token collection;
only the observer resource sampler is synthetic.
After cleanup/quiescence, bootstrap the unchanged original
definition, verify hashes/process identity and poll health plus authenticated
ready. A real request is attempted only if ready; report success only if it
actually succeeds. The parent recorded original state Ready but `ready=false`;
restoring a definition/process is **not** restoring real readiness. Cleanup,
changed-hash or quiescence failures leave explicit recovery errors rather than
loading two models. `recovery-plan.json` and `children.jsonl` support manual
identity-checked recovery after uncatchable SIGKILL, interpreter/OS failure.

Power/battery/thermal, RSS, memory-pressure and `vm_stat` are sampled throughout.
Loading/warmup swap is separate from steady requests. The observer brackets
every request directly; live steady-request swap growth is also monitored.
Stop above 512 MiB additional system swap-out or 38.5 GiB observed Metal
allocation, supply changes, policy battery/thermal limits, crashes or count/token
mismatch. No whole-cohort claim is made from incomplete coverage.

## Explicit limitations

Immutable streaming APIs expose neither full checkpoint logits nor internal
states; the controller does **not** claim them. The strict independent native
operator oracle and deterministic full-model checks remain mandatory.
Historical input token-ID manifests are pinned, but only actual input **counts**
are observable through the runtime API. Physical residency and continuous Metal
allocation high-water are unavailable; logs/terminal device-allocation samples,
RSS and memory pressure are retained, not relabeled as physical GPU peaks.
The harness has CPU/synthetic coverage, not live service/GPU acceptance.
The parent must review lifecycle behavior before granting execution exclusivity.
