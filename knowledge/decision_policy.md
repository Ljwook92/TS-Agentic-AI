# TS-SatFire Decision Policy

This policy defines the first deterministic routing rules for the planner.

## Scope

- Target tasks: `af`, `ba`, `pred`
- Available tools:
  - `dataset_gen_afba`
  - `dataset_gen_pred`
  - `run_spatial_model`
  - `run_seq_model`
  - `run_spatial_temp_model`
  - `run_spatial_temp_model_pred`
- Primary success metrics: Dice, IoU
- Retry budget: 2 automatic retries before manual review

## Baseline Policy

1. If prepared `.npy` arrays do not exist, generate them first.
2. For `af` and `ba`, start with `run_spatial_model`.
3. For `pred`, start with `run_spatial_temp_model_pred`.
4. Escalate to spatiotemporal routes when baseline quality is weak or temporal evidence matters.
5. Stop when metrics are acceptable or retries are exhausted.

## TS Adapter Rules

| Tool | Adapter Policy | Behavior |
| --- | --- | --- |
| `run_spatial_model` | `spatial_framewise` | flatten `[B,C,T,H,W]` into `[B*T,C,H,W]` |
| `run_spatial_temp_model` | `spatiotemporal_native` | keep `[B,C,T,H,W]` |
| `run_spatial_temp_model_pred` | `spatiotemporal_native` | keep `[B,C,T,H,W]` |
| `run_seq_model` | `temporal_sequence` | preserve temporal ordering for sequence models |

## Routing Rules

| Condition | Preferred Action | Why |
| --- | --- | --- |
| Task is `pred` and `FirePred` folders exist | `dataset_gen_pred` then `run_spatial_temp_model_pred` | prediction requires the 27-channel auxiliary stack |
| Task is `af` and only a fast baseline is needed | `dataset_gen_afba` then `run_spatial_model` | spatial baseline is cheaper and easier to compare |
| Task is `af` and temporal-only comparison is desired | `run_seq_model` | temporal baseline is relevant only for AF |
| Task is `ba` | `dataset_gen_afba` then `run_spatial_model` | BA uses the same derived arrays as AF but different labels |
| AF or BA baseline metrics are weak | `run_spatial_temp_model` | burned area and cumulative heat signatures benefit from temporal context |
| `FirePred` is missing for a fire | do not route to `pred` | prediction input would be incomplete |

## Parameter Adaptation Rules

| Observation | Parameter Change | Expected Effect |
| --- | --- | --- |
| CUDA OOM in 3D model | reduce `batch_size` first | lower memory pressure |
| temporal coverage seems too short | increase `ts_length` | give the model a longer history |
| windows are too sparse | reduce `interval` | sample more consecutive observations |
| training unstable or NaN loss | lower `learning_rate` | improve optimizer stability |
| dataset generation says `No enough TS` | reduce `ts_length` | allow shorter wildfire sequences |

## Hard Constraints

- Never use `run_seq_model` for `pred`.
- Never use `run_spatial_temp_model_pred` unless `FirePred` exists.
- Never assume land-cover encoding from MCD12Q1 layer names other than confirmed `LC_Type1` without verifying a sample.
- Never exceed 2 automatic retries for the same configuration class.

## Evaluator Thresholds

| Situation | Evaluator Decision |
| --- | --- |
| non-zero process exit | `needs_debug` |
| no parsed metrics but process succeeded | `needs_review` |
| spatial AF/BA baseline metric below 0.30 | `retry_with_spatiotemporal` |
| prediction run has data-shape error | `needs_debug` |
| metrics are present and above baseline floor | `complete` |

## Open Questions

- The exact export logic for `FirePred` GeoTIFFs is still external to the public repository.
- Prediction metric thresholds should be calibrated from actual prior runs, not reused from AF/BA by default.
