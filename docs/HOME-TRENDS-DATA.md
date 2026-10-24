# Homepage trends: data contract (v1)

The homepage is four linked commit charts, selected-run details and **all** planned
task rows. The previous homepage remains at `progress.html`; the full notebook
and public evidence JSON files are retained. Nothing in this feature measures
inference, deploys services or depends on a future results PR.

Build and offline acceptance from the repository root:

```sh
python3 scripts/build-progress-site.py
python3 scripts/test-home-trends.py
node scripts/test-home-trends.js
```

Open `artifacts/pages/index.html` (also works over `file://`). The build uses
Python's standard library only. JavaScript, CSS and SVG are local; there are no
runtime package or CDN dependencies. Node is only an acceptance-test dependency.
The Pages build runs these checks. A separate `site-acceptance.yml` runs the same
checks on relevant pull requests with only `contents: read`, no persisted checkout
credentials, no deployment job and no preview publication. Generated outputs are
ignored, not committed. Both workflows fetch Git history so the pinned historical
commit timestamp check works in CI as well as this checkout.

## Sources and output

`site/trends.json` is the reviewed manifest. `scripts/home_trends.py` imports
the two published Pi summaries, verifies their source/configuration and official
counts, and validates the normalized data. It writes no source files. The build
emits the same normalized object as `trends-data.json` and embeds it safely in
`index.html` for offline use.

The manifest has `schemaVersion: 1`, `commitTimeline`, `configurations`, `imports` and `runs`.
`imports` is the explicit historical Pi adapter, not a generic filename scanner.
Future collectors can add normalized records to `runs`; imported and native
records pass the same validator. Do not copy the approved sketch or acceptance
fixtures into either production input.

The normalized object has `schemaVersion: 1`, `commitTimeline`, `configurations` and `runs`.
Derived `metrics` and `totals` are recomputed during validation, never trusted
from input. Required missing numeric fields are **JSON null**, not omitted, zero,
empty string, NaN or a guessed value. Zero is valid only when actually observed.
Counts are nonnegative integers; seconds/milliseconds are finite nonnegative
numbers. A measured prefill/decode interval duration must be strictly positive.
Zero decode seconds on a zero/one-token output is allowed as a no-interval record,
but contributes no decode observation.

### Comparable configuration

Each configuration has a stable slug `id`, `label`, and:

| Fields | Meaning |
|---|---|
| `model`, `modelRevision`, `weights`, `kv` | Exact model revision and precision |
| `cache`, `chunking` | Cache policy and computational/transfer chunking |
| `topology`, `network`, `power` | Devices, layer placement, route and power conditions |
| `workload` | Harness/version, tools, concurrency, context/output limits, warmup and timeout policy |
| `datasetRevision`, `datasetSha256`, `plannedTasks` | Pinned dataset and ordered, **predeclared** task IDs |
| `timing.wall`, `timing.prefill`, `timing.decode`, `timing.ttft` | Measurement boundaries; TTFT literal below |
| `latestBaselineRun` | Explicit verified main run ID, or null if none |

Never reuse a configuration ID after changing precision, cohort, topology,
network, cache, workload or timing scope. Configurations are separate selectable
series: neither main lines nor candidate branches cross them. Environment
limitations (for example the rerun's open TUI/nonexclusive machine) also belong in
the run description. These Pi histories are observations, not controlled A/B
evidence of a speed change.

### Run identity and provenance

Run IDs and experiment IDs are stable lowercase slugs. `commitTimeline` contains
one `{sha, committedAt, evidence}` per distinct measured commit, exactly covering
the runs. `committedAt` is the verified Git **committer** timestamp in UTC
`YYYY-MM-DDTHH:mm:ssZ` form; `evidence` links to the commit. Validation rejects
missing, duplicate, extra or malformed identities/timestamps. The historical
timestamp is also checked against `git show --format=%ct` by acceptance tests.
Chronological commit axes, run controls and main lines sort by this timestamp
(full SHA breaks identical-timestamp ties), independently of input array order.
Within a commit, repeated runs sort by recorded date then stable run ID, without
inventing sub-day measurement timestamps where the source gives only a date.
Never sort by pass rate or speed.

Preserve repeated runs at one SHA: the axis has one commit tick and individually
offset markers. Each group reserves at least one target-sized slot per run, with
8 px between hit boxes (34 px fine pointer / 44 px coarse pointer). Dense groups
expand the locally scrollable chart instead of scaling labels or overlapping
targets; the whole page does not overflow. The run selector always lists every run
including failed/missing metrics.

Required run fields:

- `id`, `experimentId`, `configurationId`, `title`, `date`, `change`, `result`.
- `state`: `adopted`, `pending-review`, `regressed`, `failed` or `not-adopted`.
  This is independent of code provenance; a main rerun can regress.
- `measuredCommit`: full **actual runtime** SHA, not the harness, site, report
  or latest checkout SHA. `runtime` maps component names to `{commit: fullSHA}`.
  The current adapter requires all components on that measured SHA; mixed-runtime
  observations need a separately designed configuration/adapter.
- `binarySha256`: map of artifact names to SHA-256, possibly empty if unacquired;
  `binaryEvidence` records the source. The rerun reuses the original published
  hashes because its report explicitly states that binaries did not change.
- `provenance`: `{kind: "main", mainCommit: fullSHA, evidence: URL-or-repo-path}`
  or `{kind: "candidate", evidence: URL-or-repo-path}`. Main status is a reviewed
  assertion supported by runtime/code-adoption evidence, **never inferred** from
  where a result file is stored or a results PR being merged.
- Squash/rebase adoption keeps `measuredCommit` unchanged and records the actual
  `mainCommit` mapping. When these differ the UI says **merge revision NOT
  remeasured**. Do not relabel a measurement with the merge SHA.
- `baselineRunId`: another run in the same configuration, or null.
- `links`: `{role, label, url}` entries for known Issues, runtime code PR, notebook,
  public evidence and full report. Do not fabricate a missing results-PR link.
  Role is `issue`, `code-pr`, `results-pr`, `harness-pr`, `notebook`, `evidence`
  or `report`. Absence is shown only if no link has role `results-pr`, independent
  of its display label or language. A results link never changes code provenance.
  Historical sources do not record a separate results PR; the UI says so.
- Optional harness SHA/hash, `reportedTotals` and `rawEvidence` are provenance,
  never replacements for measured runtime identity or missing task observations.

Default selection is `latestBaselineRun` (verified means the fixed cohort's
official pass status is known, not necessarily 100% passed); otherwise the first
record in that series. Candidate-only series may have no main baseline.
Deep links use `#experiment=<experimentId>&run=<runId>`. An experiment-only
link selects its first recorded run. A run-only link is also supported. Invalid
or mismatched IDs produce a visible notice and the explicit default; hash/Back
navigation and configuration changes synchronize all four charts and details.
Chart anchors `#prefill-trend`, `#decode-trend`, `#ttft-trend`, `#task-pass-trend`
are retained. Points are native Tab/Enter/Space buttons with pressed state, full
run/SHA accessible labels, main circles, candidate diamonds/dashed branches,
a baseline ring, and a separate rectangular selection outline.

### Planned tasks, official pass rate and totals

`tasks` must contain every `plannedTasks` ID exactly once and in that order.
Each row has `taskId`, positive integer `attempt`, `wallSeconds`, `inputTokens`,
`outputTokens`, `modelRequests`, `status`, `officialValidation`, and `source`
(published JSON filename plus JSON pointer, or an evidence URL).

Status is `passed`, `failed`, `error`, `timeout`, `incomplete` or `unverified`.
`officialValidation` is `passed`, `failed`, `not-run` (explicit execution
failure/timeout/incomplete), or `unverified` (unknown assertion validation).
Completion, syntax validity, or existence of an answer is not official validation.

Pass rate is **officially passed tasks / the whole predeclared cohort**. Failed,
timed-out, errored and incomplete tasks remain in the denominator. If any official
validation is unknown, the rate is null/unverified, with confirmed passes still
visible. A no-solution execution error is a known nonpass (`not-run`), not an
executed official assertion failure; coverage distinguishes the two.

Wall time is Pi subprocess start to exit, including requests, tools, files and
scheduling. Timeout rows retain actual cutoff wall time. Input and output tokens
sum **all model requests within that task attempt**, including repeated contexts.
They are not unique prompt length/final file size. Keep retries as independently
identified attempts/runs; never replace a failed planned task with its retry.
The current historical cohorts use attempt 1; the separate HumanEval/149 retry is
described and excluded, not spliced into the cohort.

`totals[field]` contains `{value, samples, total, complete}`. Value is the sum of
known rows or null if none are known. Incomplete totals are explicitly marked
PARTIAL with row coverage, including the distinction between a known zero subtotal
and no observations. Independently `reportedTotals` may preserve authoritative
run-level totals. Complete row totals must match it (wall tolerance 0.00051 s per
rounded row); partial known subtotals cannot exceed it. Unknown rows are never
filled by dividing a run total. Complete task request counts must match the run's
request count. Complete per-task request observations must match known cumulative
task input/output counts.

### Request measurements and aggregation

Every run has `requestCount` (the whole run, including failed requests) and
`requests` (null when unacquired, otherwise an observed subset). Each request has
unique `id`, `taskId`, `attempt`, `source`, `outcome` (`completed`, `failed`,
`timeout`, `incomplete`), and all five nullable numeric fields:

```json
{
  "id": "request-id",
  "taskId": "HumanEval/156",
  "attempt": 1,
  "source": "public-evidence.json#/requests/0",
  "outcome": "completed",
  "inputTokens": null,
  "prefillSeconds": null,
  "outputTokens": null,
  "decodeSeconds": null,
  "clientTtftMs": null
}
```

Pair tokens with their **own** duration. Aggregate only pairs with both values:

- Prefill: `sum(inputTokens) / sum(prefillSeconds)`.
- Decode: `sum(outputTokens - 1) / sum(decodeSeconds)` for observations with
  **at least two output tokens**. Zero/one-token requests have no decode interval;
  exclude both their tokens and duration from this metric, even if a positive
  duration was reported. An all-zero/one-token sample has null decode speed and
  zero coverage, never fake zero throughput. Real zero input rates or client TTFT
  remain visible as zero.
- Never average per-request rates or use Pi wall time for either metric.
- Partial pairs are validated separately, but do not contribute a fake zero.
  Each metric reports paired sample count / **all** requests.
- TTFT: median of actual **client** send-start to first **valid output token**
  in milliseconds (text or tool-call content). A nonnull `clientTtftMs` requires
  `ttftBoundary: "client-send-to-first-valid-output-token"`. Role events, headers,
  heartbeats, worker ready, internal first-token time and nonstreaming total
  response latency do not qualify.
- `firstTokenReceived: false` on a failed/timeout/incomplete observation with
  null TTFT records a known failure without output. Unknown receipt is counted
  separately, not claimed as an observed no-token failure. A measured TTFT cannot
  coexist with `firstTokenReceived: false`.
- TTFT includes `samples`, `total`, `failedWithoutToken`,
  `unknownFirstTokenFailures`, `unobservedRequests`. Missing instrumentation never
  proves zero failures or full coverage. No sampling from total response time.

The validator rejects invalid numbers, zero/negative measured denominators,
duplicate identities, unknown task mappings, incompatible baseline references,
unsubstantiated pass claims, candidate-as-main-baseline, invalid hashes and unsafe
evidence URLs. Actual truth of timing/provenance still requires source review;
declaring a boundary string is not itself measurement evidence.

## Historical inventory and limitations

Both Pi runs measured **e017d84e805c016d5912b5051e4c69f758056363** on all three
runtime components. `c5913cf...` and `1571071...` identify harness history, not new
runtime measurements. The reports explicitly establish main provenance.

| Run | Official pass | Task wall rows | Task input/output rows | P/D/TTFT samples |
|---|---:|---:|---:|---:|
| `pi-main-20261023` | 20/20 | 20/20 | 20/20 | 0/98 |
| `pi-main-rerun-20261023` | 19/20 | 20/20 | 0/20 | 0/103 |

The first run's cumulative tokens are 222744 input / 19212 output. The rerun
published only run-level 241162 input / 19949 output, **not per-task tokens or
request counts**. All 20 rerun rows remain present, but those fields are null;
their task totals are incomplete. Its 9.039 s error row remains in the cohort.
Per-task wall values are rounded to milliseconds; reported run totals retain
more precision. Raw events, generated solutions and assertion logs are **local
only**; reports publish hashes, not downloadable raw logs. No additional speed
history is imported without fully established measured SHA/timing/configuration.

## Isolated synthetic acceptance fixtures

`tests/home-trends/fixtures.json` is explicitly synthetic and is never read by
the production builder. The tests transform copies of imported rows to exercise
main, candidate, regression, failure, timeout, incomplete, unverified, missing,
real zero, repeated SHA, partial cumulative totals, client TTFT median/coverage
and a separate incompatible BF16 series. The arithmetic fixture is intentionally
not the approved sketch. Generate a clearly labeled local browser page with:

```sh
python3 scripts/test-home-trends.py --fixture-output artifacts/home-trends-fixture
```

Open `artifacts/home-trends-fixture/index.html`. The command refuses the production
Pages output directory. Synthetic request source links point to its copied fixture
input; copied historical task/evidence links are context, **not evidence for the
synthetic values**. Do not deploy this directory. Node tests exercise actual render
and event handlers through a minimal offline DOM adapter; this is not a browser
layout or assistive-technology test. Human/Codex browser acceptance should check
desktop 2×2 and narrow stacked layouts, native keyboard selection, no page-level
overflow (task table scrolls locally), focus retention, deep links and Back.
