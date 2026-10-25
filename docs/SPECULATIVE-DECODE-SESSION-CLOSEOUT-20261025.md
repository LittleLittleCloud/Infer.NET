# Speculative decoding session closeout

Date: 2026-10-25 America/Los_Angeles. The user explicitly requested stopping further optimization research in this session, closing unadopted experimental code PRs, and merging research/results records so the project homepage retains the work. This is an administrative and documentation closeout, not a new experiment or a claim that performance targets passed.

## Disposition

| Work | Research | Code disposition | Results |
| --- | --- | --- | --- |
| Decode bottleneck investigation, Issue #70 | PR #71 | No implementation; approved diagnostic plan was deferred when the user prioritized speculation, then discontinued at session closeout | No new measurements; preserve source review and provenance |
| Speculative state transactions, Issue #72 | PR #73 | PR #74 closed without adoption | PR #75: exact selected state recovery passed; full service acceptance failed |
| Official Qwen3.6 MTP, Issue #76 | PR #77 | PR #78 closed without adoption | PR #79: both pilot widths failed G5; later diagnostic20 MTP arm preserved in PR #82 |
| Native DFlash2 versus MTP, Issue #80 | Existing pinned DFlash2 research and approved experiment | PR #81 closed without adoption | PR #82: DFlash2 was slower; strict draft intermediate numerical failures retained |

Research PRs #71/#73/#77 and independent results PRs #75/#79/#82 are integrated with their original commits through the closeout PR. A merge commit preserves their ancestry; do not squash this integration. Their records do not incorporate experimental inference code into main. The implementation branches and immutable tested commits remain available for later independent work.

## What was measured

All product experiments used native C#/Metal on one M5 Pro with 20 GPU cores and 48 GiB, unchanged Qwen3.6 affine8 target weights and FP32 target KV/state. MTP and DFlash2 used BF16 draft matrices. Pi 1.4.0 drove fixed HumanEval cohorts. Detailed reports retain exact runtime, model, artifact, configuration, request, failure and resource evidence. Five-task pilots and twenty-task diagnostics are separate workloads; different agent trajectories are not isolated algorithm effects.

| Experiment / workload | Main decode tok/s | Candidate result | Decision |
| --- | ---: | --- | --- |
| #72, HumanEval20 | 67.449858 | Candidate off 67.390122; synthetic on has no valid standard decode speed point. Main official/Pi 20/20; off official 20/20, Pi 19/20; on official/Pi 19/20 | No acceleration claim; full service/resource acceptance incomplete |
| #76, pilot5 | 67.138 | MTP width1 69.416; width2 68.315. Identical five-request replay gains +3.330%/+4.752%, with prefill latency regressions +8.125%/+6.284% | Neither width met the original 1.20x decode and regression/evidence gates |
| #80, diagnostic HumanEval20 | 67.781934 | MTP width1 69.562924, official/Pi 19/20; DFlash2 block4 57.980756, official/Pi 20/20; candidate off whole-cohort decode Not Obtained | DFlash2 -14.46% versus main and -16.65% versus MTP; no adoption |

The #76 plan stopped before a formal20 acceptance run. The later user-approved #80 diagnostic20 comparison includes the same frozen MTP candidate; it does not retroactively turn the failed #76 pilot gate into a pass. In #80, the off arm timed out on one task and has 114/115 terminal worker clocks. Its partial rate cannot substitute for whole-cohort decode. MTP's API failure remains in the denominator.

## Findings retained

- Selected target-state rollback and exact greedy replay checks passed. They establish useful correctness evidence, not application acceleration or complete resource acceptance.
- Official MTP runs on the Metal GPU. High draft acceptance did not recover the costs of target verification, draft-cache maintenance, synchronization and replay. Component host timers are diagnostic and do not establish per-kernel GPU attribution.
- Both DFlash2 draft numerical attempts failed the unchanged maximum-absolute-error criterion: 0.01171875 and 0.01025390625 versus the strict 0.01 limit. Passing final normalized outputs and nine separate target/state diagnostics do not erase these failures.
- Continuous Metal-driver peak evidence remains unavailable. Resource samples and whole-system swap counters are retained, but do not prove physical residency or identify GPU migration as the cause of slowdown.
- Splash's Q8 KV, draft representation, memory settings and timing scope differ. Its historical speed is useful context, not a controlled estimate of the gain from adding speculation alone.
- The final read-only Ignis review inspected upstream `gpillon/ignis@2205db6c6853089c18934ac957dbf19cf28eda1a`. Its Blackwell/NVFP4 path, recorded-state commit mechanism and profiling records suggest possible future investigations; none was implemented or benchmarked here. No new optimization experiment is scheduled by this closeout.

## Publication and baseline

The homepage retains all 22 existing and newly integrated trend runs, including failed and missing measurements, plus the research and closeout notebook entries. Successful controls do not become a newly adopted baseline merely because their results are published. Existing adopted baseline selection, tested SHAs, per-request measurements and numerical gates remain unchanged. The original service restoration evidence remains in each experiment report; this closeout changes no running inference service or machine setting.

Issues #72/#76/#80 are to be closed after the results merge and Pages verification, with the final PR dispositions recorded. Issue #70 is closed as discontinued without completing its proposed diagnostic experiment. These are experiment lifecycle decisions, not completion of the broader decoding performance target.

## Records

- [Initial decode research](DECODE-RESEARCH-SONNET-20261024.md), [DFlash2 research](DFLASH2-RESEARCH-20261024.md), [official MTP research](RESEARCH-QWEN36-OFFICIAL-MTP-20261024.md).
- [State recovery results](EXPERIMENT-72-SPECULATIVE-STATE-20261024.md), [MTP pilot results](EXPERIMENT-76-OFFICIAL-MTP-20261024.md), [DFlash2/MTP diagnostic20 results](EXPERIMENT-80-DFLASH2-RESULTS-20261024.md).
- [Homepage](https://littlelittlecloud.github.io/Infer.NET/), [experiment notebook](https://littlelittlecloud.github.io/Infer.NET/experiments.html#speculative-decode-closeout-20261025).
