# Original TS-SatFire Pred Runner

This folder keeps an original-code-style TS-SatFire `pred` runner separate from the modified TS-Agentic-AI experiment code.

Adaptations from the upstream runner are intentionally limited to:

- HPC/data paths via environment variables.
- `wandb` disabled through a dummy logger.
- Missing prepared 2021 test arrays are skipped instead of crashing.
- Evaluation plots are disabled by default; set `TS_SATFIRE_ORIG_SAVE_PLOTS=1` to save them.

Default environment variables:

```bash
TS_SATFIRE_ORIG_DATASET_ROOT=/home/jlc3q/data/SatFire/dataset/pred
TS_SATFIRE_ORIG_ROI_DIR=/home/jlc3q/New_project/TS-Agentic-AI/legacy/roi
TS_SATFIRE_ORIG_CHECKPOINT_ROOT=/home/jlc3q/data/SatFire/checkpoints/original_ts_satfire
```

Example:

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
