# TS-SatFire Tool Catalog

This document maps the preserved TS-SatFire scripts into agent-usable tools.

## `dataset_gen_afba`

- `tool_id`: `dataset_gen_afba`
- `legacy_script`: [`legacy/scripts/dataset_gen_afba.py`](/Users/lee/Github%20Clone/TS_Agentic_AI/legacy/scripts/dataset_gen_afba.py)
- `owner_role`: `executor`
- `primary_task`: `af`, `ba`
- `purpose`: build train, validation, or test `.npy` datasets from raw GeoTIFF wildfire folders
- `when_to_use`: before any AF or BA model training/testing when derived `.npy` arrays are missing
- `when_not_to_use`: when prepared arrays already exist and only model execution is needed

## Inputs

| Argument | Required | Allowed Values | Default | Meaning |
| --- | --- | --- | --- | --- |
| `mode` | yes | `train`, `val`, `test` | - | dataset split to generate |
| `ts_length` | yes | integer | - | time-series window length |
| `interval` | yes | integer | - | step between observations |
| `use_case` | yes | `af`, `ba` | - | task-specific preprocessing |

## Expected Files

- Input raw data root: `/home/jlc3q/data/SatFire/ts-satfire` by default
- ROI metadata: `legacy/roi/us_fire_*.csv`
- Output datasets: `/home/jlc3q/data/SatFire/dataset/dataset_{split}`

## Outputs

- AF/BA `.npy` image stacks
- AF/BA `.npy` label stacks
- optional test-time AF tokenized arrays

## Failure Modes

| Symptom | Likely Cause | First Fix | Second Fix |
| --- | --- | --- | --- |
| `empty file list` | missing raw fire folder or no VIIRS data | verify fire folder exists | verify `SATFIRE_ROOT` |
| no output `.npy` files | invalid split or missing ROI ids | verify ROI CSVs | inspect fire ids |

## `dataset_gen_pred`

- `tool_id`: `dataset_gen_pred`
- `legacy_script`: [`legacy/scripts/dataset_gen_pred.py`](/Users/lee/Github%20Clone/TS_Agentic_AI/legacy/scripts/dataset_gen_pred.py)
- `owner_role`: `executor`
- `primary_task`: `pred`
- `purpose`: build prediction `.npy` datasets from VIIRS and `FirePred` GeoTIFF stacks
- `when_to_use`: before prediction model training/testing when derived arrays are missing
- `when_not_to_use`: when prediction `.npy` arrays already exist

## Inputs

| Argument | Required | Allowed Values | Default | Meaning |
| --- | --- | --- | --- | --- |
| `mode` | yes | `train`, `val`, `test` | - | dataset split |
| `ts_length` | yes | integer | - | sequence length |
| `interval` | yes | integer | - | sequence stride |

## Expected Files

- Input raw data root: `/home/jlc3q/data/SatFire/ts-satfire`
- Required per-fire folders: `VIIRS_Day`, `VIIRS_Night`, `FirePred`
- Output datasets: `/home/jlc3q/data/SatFire/dataset/dataset_{split}`

## Outputs

- prediction `.npy` image stacks with 27 channels
- prediction labels representing newly burned area

## Failure Modes

| Symptom | Likely Cause | First Fix | Second Fix |
| --- | --- | --- | --- |
| missing `FirePred` path | prediction auxiliary stack absent | inspect raw dataset | drop `pred` route for that fire |
| `No enough TS` | time series too short for requested `ts_length` | reduce `ts_length` | adjust `interval` |

## `run_spatial_model`

- `tool_id`: `run_spatial_model`
- `legacy_script`: [`legacy/scripts/run_spatial_model.py`](/Users/lee/Github%20Clone/TS_Agentic_AI/legacy/scripts/run_spatial_model.py)
- `owner_role`: `executor`
- `primary_task`: `af`, `ba`
- `purpose`: train or test 2D spatial baselines on prepared `.npy` datasets
- `when_to_use`: fast baseline for AF/BA or first-stage exploration before heavier models
- `when_not_to_use`: when temporal context is clearly required or when prediction task is targeted

## Inputs

| Argument | Required | Allowed Values | Default | Meaning |
| --- | --- | --- | --- | --- |
| `model` | yes | `unet`, `attunet`, `unetr2d`, `unetr2d_half`, `swinunetr2d` | - | spatial architecture |
| `mode` | yes | `af`, `ba` | - | task |
| `batch_size` | yes | integer | - | batch size |
| `learning_rate` | yes | float | - | optimizer LR |
| `channels` | yes | integer | - | usually `8` |
| `ts_length` | yes | integer | - | used to resolve dataset filenames |
| `interval` | yes | integer | - | used to resolve dataset filenames |
| `test` | no | flag | false | run test path |

## Outputs

- training logs
- validation Dice and IoU printed to stdout
- checkpoints in `/home/jlc3q/data/SatFire/checkpoints`

## Failure Modes

| Symptom | Likely Cause | First Fix | Second Fix |
| --- | --- | --- | --- |
| file not found for dataset | missing generated `.npy` arrays | run `dataset_gen_afba` first | fix root path |
| CUDA OOM | batch size too large | lower `batch_size` | choose smaller model |
| weak Dice/IoU | spatial-only model insufficient | escalate to `run_spatial_temp_model` | increase `ts_length` |

## `run_seq_model`

- `tool_id`: `run_seq_model`
- `legacy_script`: [`legacy/scripts/run_seq_model.py`](/Users/lee/Github%20Clone/TS_Agentic_AI/legacy/scripts/run_seq_model.py)
- `owner_role`: `executor`
- `primary_task`: `af`
- `purpose`: train/test temporal sequence models on pixel time series
- `when_to_use`: active-fire experiments that focus on temporal-only reasoning
- `when_not_to_use`: burned area or prediction routing, or when TensorFlow environment is unavailable

## Inputs

| Argument | Required | Allowed Values | Default | Meaning |
| --- | --- | --- | --- | --- |
| `model` | yes | `t4fire`, `gru_custom`, `lstm_custom` | - | temporal model |
| `mode` | yes | `af` | - | task |
| `ts_length` | yes | integer | - | sequence length |
| `interval` | yes | integer | - | time spacing |
| `channels` | yes | integer | - | input channels |
| `batch_size` | yes | integer | - | batch size |

## Outputs

- TensorFlow checkpoints
- F1 and IoU summaries
- optional diagnostic plots during test mode

## Failure Modes

| Symptom | Likely Cause | First Fix | Second Fix |
| --- | --- | --- | --- |
| TensorFlow dependency errors | environment mismatch | use dedicated env | skip temporal baseline |
| poor AF performance | weak temporal-only fit | route to spatial or spatiotemporal model | tune `ts_length` |

## `run_spatial_temp_model`

- `tool_id`: `run_spatial_temp_model`
- `legacy_script`: [`legacy/scripts/run_spatial_temp_model.py`](/Users/lee/Github%20Clone/TS_Agentic_AI/legacy/scripts/run_spatial_temp_model.py)
- `owner_role`: `executor`
- `primary_task`: `af`, `ba`
- `purpose`: train/test 3D spatial-temporal models on AF/BA datacubes
- `when_to_use`: stronger AF/BA route when baseline spatial performance is insufficient
- `when_not_to_use`: when compute budget is tight and a spatial baseline is enough

## Inputs

| Argument | Required | Allowed Values | Default | Meaning |
| --- | --- | --- | --- | --- |
| `model` | yes | `unet3d`, `attunet3d`, `unetr3d`, `unetr3d_half`, `swinunetr3d` | - | spatiotemporal model |
| `mode` | yes | `af`, `ba` | - | task |
| `batch_size` | yes | integer | - | batch size |
| `learning_rate` | yes | float | - | optimizer LR |
| `channels` | yes | integer | - | usually `8` |
| `ts_length` | yes | integer | - | temporal window |
| `interval` | yes | integer | - | temporal stride |

## Outputs

- training logs
- validation Dice and IoU
- checkpoints

## Failure Modes

| Symptom | Likely Cause | First Fix | Second Fix |
| --- | --- | --- | --- |
| CUDA OOM | 3D model too heavy | lower `batch_size` | reduce `ts_length` |
| file mismatch | generated dataset does not match selected task | regenerate arrays | inspect `mode` |

## `run_spatial_temp_model_pred`

- `tool_id`: `run_spatial_temp_model_pred`
- `legacy_script`: [`legacy/scripts/run_spatial_temp_model_pred.py`](/Users/lee/Github%20Clone/TS_Agentic_AI/legacy/scripts/run_spatial_temp_model_pred.py)
- `owner_role`: `executor`
- `primary_task`: `pred`
- `purpose`: train/test prediction models using the 27-channel input stack
- `when_to_use`: next-day wildfire progression prediction
- `when_not_to_use`: if `FirePred` data or prediction `.npy` arrays are missing

## Inputs

| Argument | Required | Allowed Values | Default | Meaning |
| --- | --- | --- | --- | --- |
| `model` | yes | `unet3d`, `attunet3d`, `unetr3d`, `unetr3d_half`, `swinunetr3d`, `utae` | - | prediction model |
| `mode` | yes | `pred` | - | prediction task |
| `channels` | yes | integer | - | expected `27` before land-cover one-hot expansion in loader |
| `ts_length` | yes | integer | - | temporal window |
| `interval` | yes | integer | - | temporal stride |
| `batch_size` | yes | integer | - | batch size |
| `seed` | no | integer | `42` | random seed |

## Outputs

- prediction training logs
- validation Dice and IoU
- checkpoints and optional figures

## Failure Modes

| Symptom | Likely Cause | First Fix | Second Fix |
| --- | --- | --- | --- |
| missing `FirePred` | incomplete raw dataset | inspect fire folder | skip prediction task |
| shape mismatch | wrong channel assumption | verify 27-channel stack | inspect sample with `gdalinfo` |
| NaN loss | unstable configuration or bad batch | inspect normalization | reduce learning rate |
