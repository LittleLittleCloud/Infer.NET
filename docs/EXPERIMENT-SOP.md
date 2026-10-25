# Experiment SOP

Status: The user confirmed adoption in chat on 2026-10-23. Process approval does not equal specific experiment Issue approval; each experiment still requires separate manual approval.

## Process

`main baseline → experiment Issue → human approval → code PR → evaluation → results PR → human review → merge results → decide whether to merge code based on measured benefit → close Issue`

Each experiment uses one Issue, one code PR, and one independent results PR. Failures, regressions, no obvious gains, and interruptions all retain results; whether results are merged and whether code is merged are decided separately.

## 1. Create Experiment Issue from main

Each experiment starts from the latest main, fixing the complete baseline SHA; do not continue stacking from another unmerged experiment branch.

The Issue title must state the experiment, main changes, and expected benefits, for example:

`[Experiment] CUDA weight reuse: reduce repeated loads to improve prefill (benefit to be measured)`

The Issue body must clearly state before implementation:

- Problem, hypothesis, and baseline complete SHA and existing measurement links.
- Required code changes, involved devices, and content out of scope for this round.
- Expected prefill / decode benefits; if there is basis, provide magnitude and basis; if no basis, mark as to-be-measured.
- Evaluation configuration, question groups, main metrics, correctness and resource checks, and criteria for adopting changes. For reduced-precision hardware acceleration, distinguish operator correctness, exploratory timing eligibility, and formal quality/adoption gates according to [the hardware utilization and quality policy](QWEN36-8BIT-TARGETS.md#hardware-utilization-and-quality-policy). Register the exact algorithm, quality budget, calibration/held-out inputs and scoring before candidate acceptance evaluation; do not apply FP32 whole-model tolerances universally.
- Entry point for the first page prefill / decode chart and this experiment identifier; after data is available, fill in specific measurement point links.
- Experiment plan awaiting approval, and subsequent code PR, results PR, and evidence locations.

Do not fill in measured points on the trend chart using expected benefits. Experiments with no data yet are displayed as to-be-evaluated and can be accessed via the Issue from the experiment list next to the chart.

## 2. Implement after Issue receives manual approval

The Issue must have a traceable manual approval, recording the approver, time, and corresponding plan. Creating an Issue, automatic check passing, or agent self-approval does not count as approval.

After approval, create the `codex/experiment-<issue>-<name>` branch from main, commit the implementation, create a code PR, and link the Issue. If main has changed during the approval period, update the baseline SHA and record the differences; changes that affect assumptions or evaluation criteria must be re-approved.

Maintain candidate status until the code PR is merged. If changes to the approved experiment scope, accuracy, or decision criteria are required, update the Issue first and obtain corresponding approval.

Do not use keywords that automatically close the experiment Issue upon merging in either the code PR or the results PR, to avoid premature closure.

## 3. Evaluation

For each round of control, fix the main baseline and the complete SHA of the candidate code. Verify the actual Mac/Windows versions and artifact hashes used in the run; the Git checkout HEAD cannot substitute for the runtime version. Use a new run ID and result directory for each round.

Standard optimizations iterate on the fixed HumanEval 5 problems; when comparing with main and deciding whether to retain changes, use the fixed 20 problems. Keep the same problem set, seed, model, Pi version, precision, cache, context, and output limits, and maintain comparable power and network conditions. Competing benchmarks on the same GPU are executed serially. For an explicitly approved precision-change experiment, fix and disclose each arm's precision separately; report the overall speed/quality trade-off and use matched-precision controls for implementation-only attribution.

After obtaining complete per-request worker metrics, calculate:

- prefill token/s = total input tokens / total prefill seconds.
- decode token/s = total decode intervals / total decode seconds; each N outputs correspond to N−1 intervals.

Record both absolute values, relative changes from baseline, metric coverage, correctness, resources, failure cases, and total Pi time. Do not use Pi wall time or the Dashboard's most recent request to substitute for the full round of worker throughput.

Reduced-precision candidates may collect exploratory timing after operator correctness, request/state integrity, and resource safety checks, while full-model quality is pending. Mark these results provisional; they cannot establish adoption or formal target completion. A failed historical gate remains failed under its original plan.

Performance conclusions must consider both correctness and resource conditions; an increase in speed with incorrect output cannot be judged as an improvement. Whether the benefit justifies adoption is determined by pre-approved standards in the Issue and manual review, without adjusting thresholds after seeing results. Trade-offs, such as improved prefill but degraded decode, must be explicitly disclosed.

128K formal performance and correctness acceptance still executes [Qwen3.6 8bit performance goals](QWEN36-8BIT-TARGETS.md); HumanEval results cannot be used as a substitute. Different workloads and timing metrics are presented separately.

## 4. Create Results PR After Obtaining Data

The results PR is created from a separate branch off main, containing experiment records, public summaries, and homepage trend chart data, without depending on the candidate implementation for merging. Original large files are stored in appropriate artifact locations, accompanied by hashes and accessibility notes.

Results must include at least: Experiment Issue, Code PR, Run ID, Measurement Date, Baseline and Candidate Complete SHAs, Actual Runtime Versions and Artifact Hashes for Both Ends, Evaluation Driver Version, Configuration and Workload Identifiers, Prefill/Decode, Integrity and Correctness Status, A/B Differences, Brief Analysis, and Adoption Recommendations.

Create a results PR even for failures or interruptions, explaining the cause of failure and the data obtained. Write "Not Obtained" for missing metrics; do not fill in zeros or estimated values. Do not hide failures and valid slow rounds.

## 5. Homepage Prefill/Decode Metrics Dashboard

Add four trend charts to the homepage: prefill, decode, TTFT, and task pass rate. The x-axis represents the actual tested code commit, and the y-axes represent token/s, token/s, milliseconds, and percentage, respectively; arranged by commit time, displaying short SHAs while retaining complete SHAs. Commits recorded in results or published on the site are not treated as tested code commits.

The homepage body retains only the four trend charts, along with metadata, change overview, results overview, and per-task details displayed when selecting an experiment point; the original homepage's progress, topology, target table, and experiment cards no longer occupy the homepage. Complete experiment notes and evidence remain accessible from the selected point.

TTFT is defined as the time from when the client sends the model request to when the first valid model output token is actually received; the trend chart displays the request median and discloses collection coverage. Role events, heartbeats, and worker internal ready states cannot substitute for the first token. Task pass rate is calculated as the number of officially per-problem assertion passes divided by the total number of fixed problems; timeouts, errors, and incomplete tasks are not removed from the denominator; if no assertion verification exists, it is marked as unverified. When historical reliable metrics are missing, display "Not Obtained."

Per-task details display task ID, end-to-end latency, prefill tokens, generate tokens, and result status. Latency covers the entire Pi task; token counts accumulate all model requests for that problem, including repeated contexts. Record failure, timeout, and retry identities; do not fill in zeros for missing measurements, and ensure totals can be reconciled with original records.

- Clearly mark tested code points on main; the currently adopted baseline is additionally highlighted; candidates and unadopted experiments use different styles and text labels.
- Merging a results PR into main only indicates that experiment records are saved; it does not allow marking candidate code as the main baseline.
- If squash/rebase merging is used, record the correspondence between the tested SHA and the actual code merge SHA; if the merge artifact has not been re-measured, explicitly mark it, and do not fabricate measured points for that commit.
- Each measurement point is linked to the Experiment Issue, Code PR, Results PR, Configuration, Results Analysis, and Evidence; the Issue provides a reverse link to the corresponding point on the homepage. Point links use stable experiment identifiers and run IDs.
- Multiple runs of the same commit are all retained; do not select only the fastest value. Different precisions, problem sets, networks, topologies, or timing ranges use independent series or filters, with comparable configurations displayed by default.
- Points with failures but reliable measurement values retain the failure status; points without reliable values are only displayed in the associated list, not plotted as zero.

Historical data only backfills verifiable tested SHAs and metrics. Historical results with only total Pi time cannot derive prefill/decode; records without source code versions are retained in the historical experiment area.

The above are requirements to be implemented on the homepage and do not imply that the current site already possesses trend charts.

## 6. Manual Review, Merge, and Closure

After the experiment ends, manually review the results. Regardless of whether the results PR is positive, negative, or a failure, correct any record issues and always merge into main after passing review.

When a confirmed improvement exists, the Code PR is merged into main after manual review and passing checks, updating the baseline; if there is no benefit, degradation, or insufficient evidence, the Code PR is not merged, the reason is recorded, and the candidate PR is closed. Manual review is a necessary step; this SOP does not grant agents automatic merge permissions.

If main has incorporated other changes affecting performance during the experiment, re-compare with the current main before adopting the candidate; retain original measurements and append re-measurements. After code merging, verify the actual merged version, update the baseline identity, and perform supplementary measurements if necessary, writing results back to the same experiment.

Finally, verify that results are merged, the code adoption decision is implemented, homepage and evidence links are correct, and baseline status is accurate. Write the conclusion and the disposition of both PRs in the Issue, then close the Issue.
