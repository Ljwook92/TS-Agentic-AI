# Original-Style VIIRS-Only Pred Baseline: ts=6, interval=1, best-val

## Run Identity

- Date recorded: 2026-05-21
- Task: `pred`
- Model: original TS-SatFire `SwinUNETR-3D`
- Input type: VIIRS/FirePred only, no GOES input
- Runner: `legacy/original_ts_satfire/run_spatial_temp_model_pred_original.py`
- Dataloader: `legacy/original_ts_satfire/data_generator_pred_torch_original.py`
- Metric style: original TS-SatFire-style test metric with `zero_division=1.0`
- Checkpoint selection: best validation loss

## Command

```bash
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
- Loaded best-val checkpoint: `/home/jlc3q/data/SatFire/checkpoints/original_ts_satfire/model_swinunetr3d_mode_pred_num_heads_2_hidden_size_24_batchsize_1_checkpoint_epoch_10_nc_43_ts_6.pth`
- Invalid-value cleanup: enabled

## Aggregate Test Result

- Original-style Test F1: `0.33120386589713374`
- Original-style Test IoU: `0.3059381050684117`
- Evaluated IDs: `24`
- Skipped missing IDs: `0`

## Per-Fire Test Results Captured From Log

| Fire ID | IoU | F1 |
|---|---:|---:|
| US_2021_MT4714310953420211004 | 0.6666666666666666 | 0.6666666666666666 |
| US_2021_ID4558511544420210705 | 0.204729801783861 | 0.219897144805233 |
| US_2021_ID4663811466720210707 | 0.0339848667016149 | 0.05624215090430749 |
| US_2021_MT4568311385420210708 | 0.2647058823529412 | 0.2647058823529412 |
| US_2021_ID4453211532920210810 | 0.1509433962264151 | 0.1509433962264151 |
| US_2021_ID4762711608320210708 | 0.4423076923076923 | 0.4423076923076923 |
| US_2021_WA4879111827120210805 | 0.4 | 0.4 |
| US_2021_WA4828511853120210713 | 0.1986200241856403 | 0.21894858533996248 |
| US_2021_WA4856812048820210708 | 0.0 | 0.0 |
| US_2021_WA4877811903420210803 | 0.16279069767441862 | 0.16279069767441862 |
| US_2021_MT4579011310120210708 | 0.16509273600656987 | 0.1685831539088971 |
| US_2021_CA3568711855020210818 | 0.2367859017087809 | 0.2717905700519529 |
| US_2021_CA3604711863120210910 | 0.17633958661597487 | 0.2917087395851864 |
| US_2021_CA3627811855020210815 | 0.1893437585599417 | 0.29505321562635534 |
| US_2021_CA3658211879520210912 | 0.16666221458150837 | 0.21906288846247993 |
| US_2021_CA4086312235520210630 | 0.05846262846808699 | 0.10837264624054074 |
| US_2021_NM3344410803520210514 | 0.08529465346387022 | 0.14274198446726977 |
| US_2021_CA3451712013120211011 | 0.0428219302307633 | 0.07105841018884497 |
| US_2021_AZ3368910927620210616 | 0.5274390086457253 | 0.5484199390940144 |
| US_2021_AZ3345510938920210616 | 0.454544204981531 | 0.47400990612538113 |
| US_2021_NM3676810505920211120 | 1.0 | 1.0 |
| US_2021_NM3323810847220210520 | 0.08997887047987851 | 0.15058911150265017 |
| US_2021_NM3340210587120210426 | 1.0 | 1.0 |
| US_2021_FL2521008104520210308 | 0.625 | 0.625 |

## Current GOES Comparison Reference

- GOES bottleneck best-val result previously observed:
- F1: `0.3237576247714002`
- IoU: `0.30715219912708724`
- GOES minus baseline F1: `-0.00744624112573354`
- GOES minus baseline IoU: `+0.00121409405867554`

## Notes

- This is the current fair ts=6 VIIRS-only baseline under original-style metrics and best-val checkpoint selection.
- Next ablations should test shorter VIIRS inputs, especially `ts=2` and `ts=4`, because the project hypothesis is that GOES high-frequency dynamics help when VIIRS temporal context is shorter.
