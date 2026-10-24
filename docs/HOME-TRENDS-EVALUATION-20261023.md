# Homepage four-trend charts and per-task details: Issue #44 evaluation

Status: Local evaluation passed; code and results await user review, not yet merged or published.

This experiment was approved by the user in chat. Implementation was completed by GitHub Copilot CLI 1.0.103, `gpt-6.1-sol`; Codex independently reviewed and ran browser acceptance. Code and results use two branches independently established from main, both awaiting human user review; not merged, not published, Issue not closed.

## Scope and Baseline

- Relates to [Issue #44](https://github.com/LittleLittleCloud/Infer.NET/issues/44) and [Code PR #46](https://github.com/LittleLittleCloud/Infer.NET/pull/46).
- Website baseline main: `58bf95018975297ba6f2f965948e2c5f4944cd95`; after the original proposal `630ce55`, main merged #45, which only converged model support and did not change the homepage goal.
- After new deployment records were added to main during implementation, the code was rebased to `a5f7bc79e2d9eab308717722c52f859bb92fe2de`; the accepted code is `37c93b63e6dd0eb2780d7a67989b51dbba601475`, and the build retains 44 experimental notes. The baseline screenshot saves the experimental starting `58bf950` version.
- Control: The old homepage included progress, topology, goal, and experiment cards; the candidate homepage includes four trend charts, linked experiment details, and all task details. Old content is retained in `progress.html`; complete experimental notes and public evidence are not deleted.
- This experiment only changes display and data organization; it does not update inference services, redeploy models, run GPU benchmarks, or claim increased inference throughput.

## Historical Data Verification

The source code for the retained Pi rerun is `e017d84e805c016d5912b5051e4c69f758056363`; the commits for the webpage, evaluation driver, and result records do not replace this run version.

| Run | Official per-question pass rate | End-to-end total time per question | Input tokens | Generated tokens | Per-question token data |
|---|---:|---:|---:|---:|---|
| Same-version rerun | 19/20, 95% | 774.288364666 s | 241162 | 19949 | Not published, retain null |

Input count accumulates all model requests per question, including repeated context. Summing the per-question times (rounded) may have minor differences from the original timing for the full round. The tool call error for question 5 in the HumanEval/149 rerun remains in the denominator of the fixed 20 question; independent retries for single questions are not mixed into this round's results.

This round has no qualified per-request worker prefill/decode timing or client first-token measurements, so the production homepage explicitly displays "Not Obtained" for P/D and TTFT, does not draw zero points, and does not derive speed from Pi wall time. TTFT requires the client request to be sent until the first valid output token; worker ready time cannot be used as a substitute. Existing fixed-length inference experiments are not mixed into the Pi workload.

Evidence: [Rerun Report](PI-HUMANEVAL-RERUN-20261023.md), [Rerun Summary](../site/pi-humaneval-rerun-summary.json).

## Acceptance Method and Results

Data acceptance verifies every task and overall statistics in the original summary; offline interactive checks cover deep links, candidate identity, same SHA rerun, missing and zero values, failure denominators, and configuration isolation. Independent Chromium acceptance covers production data and clearly labeled synthetic fixtures; the two are strictly isolated, and synthetic values are not published as measurement results.

The first round of review required corrections to the decode semantics for single-token outputs, commit ordering, same-commit point selection area, result PR link status, and adding CI for the read-only site before merge; mobile coordinate text was also checked via browser. The above issues have been corrected, and corresponding regression checks have been added.

Codex independently executed on the final code:

| Check | Actual Result |
|---|---|
| `python3 scripts/build-progress-site.py` | Passed: 3 page, 44 note articles, 2 real runs |
| `python3 scripts/test-home-trends.py --fixture-output artifacts/home-trends-fixture` | 18 items passed |
| `node scripts/test-home-trends.js` | 15 items passed |
| `node --check site/trends.js`, `git diff --check` | Passed |
| Playwright 1.64.1 / Chromium 149.0.7827.55 | 18 browser acceptance items passed; no runtime errors, no resource loading errors |

Browser acceptance covers 1280, 768, 390, 320px; checks four-image and per-question linkage, keyboard operations, page back, deep link restoration, all 20 questions, indistinguishable missing vs. real, official assertion denominator, independent runs with identical SHA, and BF16/FP32 series isolation. In the synthetic fixture, P/D/TTFT values only verify interaction and are not actual measurements. CLI additionally verifies 34px mouse and 44px touch click areas under 320px; final data and JavaScript tests retain this regression.

Public review entry:

- [Per-browser results](evaluations/issue-44/browser-results.json), [Build and offline check records](evaluations/issue-44/final-checks.json), [Version and hash manifest](evaluations/issue-44/manifest.json).
- [Two-branch integration check](evaluations/issue-44/integration-results.json): Merge had no conflicts; result branch built separately into 2 page, 45 note articles; combined site built into 3 page, 45 note articles; 18 Python and 15 JavaScript checks still passed. Result records can be merged independently without requiring the code PR to be merged.
- [Independent browser acceptance script](evaluations/issue-44/browser-acceptance.cjs). Build production pages and synthetic fixtures in the code PR checkout, served via local HTTP; set `PLAYWRIGHT_MODULE` (absolute path to local Playwright package), `CHROMIUM_PATH`, `PRODUCTION_URL`, `FIXTURE_URL`, `EVAL_OUTPUT` (new local artifacts directory), then run `node <path-to-this-script>`. The script does not download browsers or connect to inference services.
- [Old homepage baseline screenshot](evaluations/issue-44/baseline-main-desktop.png), [Final desktop homepage](evaluations/issue-44/desktop-overview.png), [Per-task details](evaluations/issue-44/task-detail.png), [320px homepage](evaluations/issue-44/mobile-overview.png).

CLI session, complete raw output, usage metadata, first-round checks, and review feedback are retained only in `artifacts/issue-44/`, which is ignored by development machines; the public manifest saves filenames and SHA-256; these are not public download links. Usage records confirm the actual model is `gpt-6.1-sol`. No tokens, credentials, or private prompts are disclosed.

## Adoption Recommendations and Limitations

Adoption of this homepage implementation is recommended: it meets acceptance requirements for four images, linked details, per-task breakdowns, traceable data, and handling of missing tests. The pass rate only reflects a fixed small sample of 20 questions, not the full 164 HumanEval leaderboard results; this experiment does not perform missing inference measurements, nor does it treat interface acceptance as formal 128K performance acceptance.

Currently, there is only one tested run commit and two Pi runs, so task pass rates reflect only two independent points from the same commit; the three speed/latency charts are empty. This is the current state of evidence, not a rendering fault. Subsequent result PRs can add new runs according to the code PR's data contract, retaining actual commits and configurations; unmerged candidates are not promoted to main.

Result PRs are saved after human review per SOP; code PRs are decided for merge by user review. Neither PR uses keywords to auto-close the experimental Issue; the Issue will be manually closed after merge decisions, homepage status, and evidence links are verified.

The archived `baseline-browser.json` retains the page title and headings captured from the original Chinese page. Those strings are historical browser evidence; the current site and this report use English.
