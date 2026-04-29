# VIIRS-Only Pred Baseline: ts=6, interval=1, epoch=10

## Run Identity

- Date recorded: 2026-04-29
- Task: `pred`
- Model: original TS-SatFire `SwinUNETR-3D`
- Input type: VIIRS/FirePred only, no GOES input
- Purpose: baseline result to compare against GOES temporal module experiments

## Command

```bash
PYTHONPATH=/home/jlc3q/New_project/TS-Agentic-AI/legacy \
python legacy/scripts/run_spatial_temp_model_pred.py \
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

## Dataset

- Dataset root: `/home/jlc3q/data/SatFire/dataset/pred`
- Train VIIRS: `(1519, 27, 6, 256, 256)`
- Train dataloader input after landcover one-hot: `[B, 43, 6, 256, 256]`
- Val VIIRS: `(200, 27, 6, 256, 256)`
- Val dataloader input after landcover one-hot: `[B, 43, 6, 256, 256]`
- Test fires: `24`
- Test empty-label fires skipped from aggregate: `3`

## Aggregate Test Result

- Test F1: `0.058888701222301956`
- Test IoU: `0.035171418253293965`
- Total predicted positive pixels: `60231`
- Total label positive pixels: `43926`
- Total frames evaluated: `665`
- Skipped empty-label IDs: `3`

## Per-Fire Test Results Captured From Log

| Fire ID | IoU | F1 | Pred positive pixels | Label positive pixels | Zero prediction frames |
|---|---:|---:|---:|---:|---:|
| US_2021_WA4879111827120210805 | 0.0 | 0.0 | 0 | 338 | 20/20 |
| US_2021_WA4828511853120210713 | 0.030183416316708137 | 0.04989460958731614 | 2311 | 3350 | 44/60 |
| US_2021_WA4856812048820210708 | 0.0 | 0.0 | 0 | 5107 | 31/31 |
| US_2021_WA4877811903420210803 | 0.0 | 0.0 | 0 | 4155 | 43/43 |
| US_2021_MT4579011310120210708 | 0.0053449822184128 | 0.009301858546025487 | 329 | 2538 | 78/81 |
| US_2021_CA3568711855020210818 | 0.05263840685176892 | 0.08827886085626911 | 829 | 431 | 8/16 |
| US_2021_CA3604711863120210910 | 0.15938747169615083 | 0.26625281414856505 | 18595 | 5169 | 0/20 |
| US_2021_CA3627811855020210815 | 0.1624316446356964 | 0.26028367890624626 | 776 | 347 | 1/13 |
| US_2021_CA3658211879520210912 | 0.07131674524402819 | 0.12008050130209505 | 28495 | 3609 | 3/45 |
| US_2021_CA4086312235520210630 | 0.03351633732957188 | 0.0642722638890256 | 458 | 268 | 0/4 |
| US_2021_NM3344410803520210514 | 0.06337696291105413 | 0.10972904405009931 | 557 | 1549 | 3/17 |
| US_2021_CA3451712013120211011 | 0.04911480355067626 | 0.07646973530073913 | 769 | 797 | 2/6 |
| US_2021_AZ3368910927620210616 | 0.0 | 0.0 | 160 | 877 | 11/16 |
| US_2021_AZ3345510938920210616 | 0.0 | 0.0 | 154 | 890 | 10/14 |
| US_2021_NM3676810505920211120 | skipped | skipped | 54 | 0 | 18/23 |
| US_2021_NM3323810847220210520 | 0.06827548292129851 | 0.11628966862957904 | 2736 | 3357 | 11/32 |
| US_2021_NM3340210587120210426 | skipped | skipped | 0 | 0 | 4/4 |
| US_2021_FL2521008104520210308 | 0.0 | 0.0 | 313 | 261 | 4/8 |

## Notes

- This is a VIIRS-only baseline. The GOES subdaily tensors were not loaded by this script.
- The result is useful as an early baseline/smoke baseline, but not necessarily the final paper baseline because it used only 10 epochs.
- Train/validation loss history and checkpoint path were not captured in the provided log.
- Several fires have zero-prediction behavior, so longer training or threshold/calibration checks may be needed before treating this as the official baseline.
