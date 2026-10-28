# Experiment #103 evaluation harness

**Prepared, not executed.** Only CPU/synthetic tests and compilation have run.
This controller never builds packages, changes power/sysctl, edits LaunchAgents,
or controls the original coordinator/Windows. It is not 128K acceptance.

The only additional product correction is in `SplitService.Profile`: preserve
the managed `DEEPINFER_METAL_LMHEAD` selector while clearing other experimental
environment settings. Without this, `split-serve` would erase candidate-on.
The parent must freeze/repackage this correction before running; the original
`a13abd9c5b3dc37a74e39cbbf82fcb3eda0f92ba` package is insufficient. No kernel
retuning, precision, model, protocol or formatter change is included.

## Preparation and commands

Run from the repository root. Replace uppercase placeholders with verified
absolute paths/full hashes after the parent freezes the code. Every invocation
needs a **new, nonexisting** output directory under this checkout's ignored
`artifacts/`. Files use restricted permissions; configurations, request artifacts,
Pi output and recovery commands remain local-only. Never publish raw directories.

```sh
# Safe: CPU only, owned Python children/loopback fakes; no real services, Docker, model or GPU.
python3 scripts/test-lmhead-evaluation.py -v
python3 -m py_compile scripts/issue103_common.py scripts/*lmhead*.py
dotnet build DeepInfer.slnx --no-restore

# Read-only preflight; never pauses the worker or submits model/GPU work.
# Executes package "version", Pi "--version", Docker image inspect, health/idle
# reads, file hashing and read-only resource/power queries.
python3 scripts/run-lmhead-evaluation.py \
  --phase preflight \
  --candidate-package CANDIDATE_RELEASE/deepinfer \
  --candidate-sha CANDIDATE_FULL_SHA \
  --candidate-source FROZEN_CANDIDATE_CHECKOUT \
  --output artifacts/dual-host/issue103-lmhead-affine8/preflight-UNIQUE
```

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
The user explicitly authorized **one attempt** with a full battery at 35W:
select `--power-policy battery-assisted-35w` for initial AC35W and SOC >=99%,
unchanged adapter identity and AC35W throughout, SOC >=80%, and at most a
20-percentage-point drop from admission. This is exploratory, **not stable-140W
acceptance**. It does not authorize retries. Both policies retain the fresh
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
hops. Standard summary inputs are explicitly labeled
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
