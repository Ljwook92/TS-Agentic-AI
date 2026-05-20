# Original-Style VIIRS-Only Pred Baseline: ts=6, interval=1, epoch=10

## Run Identity

- Date recorded: 2026-05-20
- Task: `pred`
- Model: original TS-SatFire `SwinUNETR-3D`
- Input type: VIIRS/FirePred only, no GOES input
- Runner: `legacy/original_ts_satfire/run_spatial_temp_model_pred_original.py`
- Dataloader: `legacy/original_ts_satfire/data_generator_pred_torch_original.py`
- Metric style: original TS-SatFire-style test metric with `zero_division=1.0`
- Purpose: original-code-style baseline for comparison with GOES temporal module experiments

## Command

```bash
PYTHONPATH=/home/jlc3q/New_project/TS-Agentic-AI/legacy \
python legacy/original_ts_satfire/run_spatial_temp_model_pred_original.py \
  -m swinunetr3d \
  -mode pred \
  -b 1 \
  -r 0 \
  -lr 1e-4 \
  -nh 2 \
  -ed 24 \
  -nc 43 \
  -ts 6 \
  -it 1 \
  -epochs 10 \
  -seed 42
```

## Environment

- Dataset root: `/home/jlc3q/data/SatFire/dataset/pred`
- ROI dir: `/home/jlc3q/New_project/TS-Agentic-AI/legacy/roi`
- Checkpoint root: `/home/jlc3q/data/SatFire/checkpoints/original_ts_satfire`
- Checkpoint: `/home/jlc3q/data/SatFire/checkpoints/original_ts_satfire/model_swinunetr3d_mode_pred_num_heads_2_hidden_size_24_batchsize_1_checkpoint_epoch_10_nc_43_ts_6.pth`
- Invalid-value cleanup: enabled in original dataloader compatibility layer

## Dataset

- Train raw VIIRS: `(1519, 27, 6, 256, 256)`
- Train dataloader input after landcover one-hot: `[B, 43, 6, 256, 256]`
- Val raw VIIRS: `(200, 27, 6, 256, 256)`
- Val dataloader input after landcover one-hot: `[B, 43, 6, 256, 256]`
- Test fires evaluated: `24`
- Missing test fires skipped: `0`

## Aggregate Test Result

- Original-style Test F1: `0.3234047255869624`
- Original-style Test IoU: `0.3036144121246314`
- Evaluated IDs: `24`
- Skipped missing IDs: `0`

## Per-Fire Test Results Captured From Log

| Fire ID | IoU | F1 |
|---|---:|---:|
| US_2021_MT4714310953420211004 | 0.7222222222222222 | 0.7222222222222222 |
| US_2021_ID4558511544420210705 | 0.20211209026784305 | 0.2162774768562497 |
| US_2021_ID4663811466720210707 | 0.03517156862745098 | 0.058116707598019124 |
| US_2021_MT4568311385420210708 | 0.2647058823529412 | 0.2647058823529412 |
| US_2021_ID4453211532920210810 | 0.1509433962264151 | 0.1509433962264151 |
| US_2021_ID4762711608320210708 | 0.46153846153846156 | 0.46153846153846156 |
| US_2021_WA4879111827120210805 | 0.4 | 0.4 |
| US_2021_WA4828511853120210713 | 0.19681575381782418 | 0.21548025302385068 |
| US_2021_WA4856812048820210708 | 0.0 | 0.0 |
| US_2021_WA4877811903420210803 | 0.16279069767441862 | 0.16279069767441862 |
| US_2021_MT4579011310120210708 | 0.16579157269713818 | 0.1696716762690703 |
| US_2021_CA3568711855020210818 | 0.250233322766336 | 0.28640967806004525 |
| US_2021_CA3604711863120210910 | 0.17661530489522798 | 0.2898513184968146 |
| US_2021_CA3627811855020210815 | 0.1507430709377777 | 0.24562483606402574 |
| US_2021_CA3658211879520210912 | 0.231741212667771 | 0.2830866486772365 |
| US_2021_CA4086312235520210630 | 0.06240424090019018 | 0.11396083867613932 |
| US_2021_NM3344410803520210514 | 0.007977290629307677 | 0.01462859391248138 |
| US_2021_CA3451712013120211011 | 0.047001153402537486 | 0.07332433648223122 |
| US_2021_AZ3368910927620210616 | 0.50534176862212 | 0.5102703088922885 |
| US_2021_AZ3345510938920210616 | 0.43416472565432535 | 0.4392560211589616 |
| US_2021_NM3676810505920211120 | 1.0 | 1.0 |
| US_2021_NM3323810847220210520 | 0.03343215509084598 | 0.058554059905225365 |
| US_2021_NM3340210587120210426 | 1.0 | 1.0 |
| US_2021_FL2521008104520210308 | 0.625 | 0.625 |

## Notes

- This result uses the original-style `zero_division=1.0` behavior. Empty-label/empty-prediction frames can receive perfect scores.
- This baseline is the correct comparison target if matching upstream TS-SatFire evaluation behavior.
- For stricter progression evaluation, compare separately against the strict metric record in `memory/pred_viirs_only_baseline_ts6_epoch10.md`.
