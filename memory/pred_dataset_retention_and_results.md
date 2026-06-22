# Prediction Dataset Retention and Result Ledger

## Current Decision

The target retained candidate dataset, after the current generation and verification complete, is the H4 partial-history, maximum-radius enriched source:

```text
pred_event_candidates_{train,val,test}_conn8_r12p0_mincomp1_h4_partial_goes_frp_motion_recent_firepred.csv
```

This dataset contains the geometry, cumulative VIIRS history, daily GOES FRP aggregates, subdaily and recent GOES motion, weather, fuel, terrain, and land-cover features. Rows can be filtered by `distance_px` to reproduce fixed candidate radii 5, 8, 10, and 12 without retaining four copies.

The local-spread experiment uses `R_local=5` only to separate connected/local spread from remote ignition components. The candidate radius remains an independent model setting. For radius comparisons, the classifier target must use `--target-scope local` so remote ignitions are not positive training labels.

The previous r5 H4-partial final dataset remains only until the r12-source sweep reproduces the r5 result and all train/val/test row counts are verified.

## Historical Dense-Model Results

These runs use the original-style whole-time-series test aggregation reported by the corresponding runner. Absolute differences are shown in percentage points, not relative percentages.

| VIIRS history | Configuration | Test F1 | Test IoU | Outcome |
| --- | --- | ---: | ---: | --- |
| 2 days | VIIRS-only SwinUNETR | 0.246567 | 0.221690 | Baseline |
| 2 days | GOES spatial frontbuf, direct fusion | 0.305479 | 0.279523 | +0.058912 F1, +0.057833 IoU over baseline |
| 2 days | GOES decoder gate | 0.279751 | 0.253238 | Below direct fusion |
| 2 days | GOES residual, buffer experiment | 0.239596 | 0.213336 | Rejected |
| 2 days | GOES residual, alternate buffer | 0.303450 | 0.277322 | Near direct fusion but not better |
| 4 days | VIIRS-only SwinUNETR | 0.274567 | 0.252403 | Baseline |
| 4 days | GOES spatial frontbuf, direct fusion | 0.305589 | 0.286474 | +0.031022 F1, +0.034071 IoU over baseline |
| 4 days | GOES residual, buffer 8 | 0.236406 | 0.212453 | Rejected |
| 4 days | GOES residual, buffer 16 | 0.278391 | 0.252994 | No meaningful gain |
| 6 days | GOES spatial frontbuf, direct fusion | 0.265550 | 0.244703 | Worse than shorter-history GOES runs |

The separately recorded ts=6 best-validation baseline in `memory/pred_viirs_only_original_metric_ts6_bestval.md` used a different run/checkpoint context and should not be merged into the table above without reproducing both sides under one protocol.

## Candidate-Model Milestones

Metrics below distinguish candidate-supported mask scores from full/local growth scores. They are not directly interchangeable with the original dense-model whole-time-series metric.

| Experiment | Test result | Interpretation |
| --- | --- | --- |
| Geometry-only classifier | PR-AUC 0.085329 | Candidate geometry alone was insufficient |
| Geometry + daily GOES FRP | PR-AUC 0.226513 | GOES FRP strongly improved ranking |
| H2 geometry + GOES FRP | mean IoU 0.323042; fire macro IoU 0.237895; fire micro IoU 0.235051 | History features substantially improved reconstruction |
| H4 GOES recent motion | test PR-AUC 0.351249; fire micro IoU 0.249018 | Recent GOES motion improved ranking and reconstruction modestly |
| H4 recent fuel/weather | test PR-AUC 0.352848; fire macro IoU about 0.215977; fire micro IoU 0.256231 | Best observed H4 micro IoU among the recorded feature ablations |
| H4 recent full FirePred | test PR-AUC 0.356778; fire macro IoU about 0.215984; fire micro IoU 0.255372 | Retained feature-complete dataset; supports all sub-ablations |
| H4 full motion + FirePred | fire macro IoU 0.214535; fire micro IoU 0.253921 | Reference for H4 versus H6 paired comparison |
| H6 full motion + FirePred | fire macro IoU 0.217236; fire micro IoU 0.255947 | Only +0.002701 macro IoU over H4; confidence interval crossed zero |

The H4 versus H6 paired bootstrap comparison gave macro IoU delta `+0.002701`, 95% CI `[-0.000523, 0.006164]`, and Wilcoxon `p=0.167651`. H6 was therefore not retained as the primary history length.

## Local and Remote Growth Diagnosis

With `R_local=5`, remote ignitions represented approximately 4.28% of test growth pixels. Excluding them improved interpretation but did not explain most of the remaining error.

| H4 feature set | Fire local-positive macro IoU | Fire local-positive macro F1 | Fire local micro IoU | Local candidate coverage micro |
| --- | ---: | ---: | ---: | ---: |
| GOES motion | 0.198436 | 0.323480 | 0.228489 | 0.842905 |
| GOES recent fuel/weather | 0.205936 | 0.333569 | 0.236268 | 0.842905 |
| GOES recent full FirePred | 0.205632 | 0.333088 | 0.236424 | 0.842905 |

The main remaining failure modes were:

1. Connected local growth extended outside the fixed radius-5 candidate domain.
2. Some dates had complete candidate coverage but probabilities remained below threshold.
3. Remote new components required a separate spotting/new-ignition task rather than deletion from the dataset.

## Candidate-Radius Oracle Results

### Validation

| Policy | Local oracle micro IoU | Candidate cost versus r5 | Local positive prevalence |
| --- | ---: | ---: | ---: |
| fixed r5 | 0.617670 | 1.000000 | 0.039558 |
| fixed r8 | 0.741457 | 1.513525 | 0.031374 |
| fixed r10 | 0.799466 | 1.920508 | 0.026660 |
| fixed r12 | 0.835572 | 2.288977 | 0.023379 |
| adaptive previous p95 | 0.730523 | 1.284728 | 0.036417 |
| adaptive history p95 max | 0.760266 | 1.579747 | 0.030822 |

### Test

| Policy | Local oracle micro IoU | Candidate cost versus r5 | Local positive prevalence |
| --- | ---: | ---: | ---: |
| fixed r5 | 0.842905 | 1.000000 | 0.025103 |
| fixed r8 | 0.917500 | 1.766450 | 0.015469 |
| fixed r10 | 0.943913 | 2.409385 | 0.011667 |
| fixed r12 | 0.960321 | 2.998490 | 0.009538 |
| adaptive previous p95 | 0.889367 | 1.142663 | 0.023180 |
| adaptive history p95 max | 0.903974 | 1.341980 | 0.020061 |

The oracle result favors adaptive history-p95 radius as the coverage/cost compromise. The retained r12 dataset is an evaluation superset, not a decision to deploy fixed radius 12.

## Deletion Policy

Preserve:

- Raw `/home/jlc3q/data/SatFire/ts-satfire` inputs.
- Raw `/home/jlc3q/data/GOES_clipped_tif_common_wgs84` inputs.
- Final H4-partial r12 GOES-recent-motion + FirePred CSVs.
- Checkpoints needed for reported baselines.
- Compact evaluation CSVs and generated Markdown archives.

Delete after archive and verification:

- H1, H2, H4 non-partial, and H6 large candidate datasets.
- Superseded `goes_frp`, `goes_frp_motion`, and non-recent FirePred intermediate CSVs.
- r12 raw and GOES-only intermediate CSVs after final r12 FirePred files have matching row counts.
- Previous r5 enriched candidate CSVs after the r12-source r5 evaluation is reproduced.
- Dense frontbuf buffer variants that are no longer part of the selected approach.
- Merged dataset `*_part_*.npy` files after confirming their final merged `.npy` exists.

Do not delete files solely from filename patterns while a generator or join process is running.
