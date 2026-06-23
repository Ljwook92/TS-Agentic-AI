#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${TS_SATFIRE_CODE_ROOT:-/home/jlc3q/New_project/TS-Agentic-AI}"
DATASET_ROOT="${TS_SATFIRE_ORIG_DATASET_ROOT:-/home/jlc3q/data/SatFire/dataset/pred}"
ROI_DIR="${TS_SATFIRE_ORIG_ROI_DIR:-$CODE_ROOT/legacy/roi}"
CHECKPOINT_BASE="${TS_SATFIRE_PAPER_CHECKPOINT_ROOT:-/local/scratch/$USER/checkpoints/ts_satfire_paper_pred}"

TS="${TS:-6}"
INTERVAL="${INTERVAL:-1}"
EPOCHS="${EPOCHS:-200}"
MODELS="${MODELS:-unet3d attunet unetr3d swinunetr3d}"
SEEDS="${SEEDS:-42}"
GOES_VARIANT="${GOES_VARIANT:-none}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-viirs43}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-val_loss}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_PROBABILITY_MAPS="${SAVE_PROBABILITY_MAPS:-1}"

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/legacy${PYTHONPATH:+:$PYTHONPATH}"
export TS_SATFIRE_ORIG_DATASET_ROOT="$DATASET_ROOT"
export TS_SATFIRE_ORIG_ROI_DIR="$ROI_DIR"
export TS_SATFIRE_ORIG_CLEAN_INVALID="${TS_SATFIRE_ORIG_CLEAN_INVALID:-1}"

required=(
  "$DATASET_ROOT/dataset_train/pred_train_img_seqtoseq_alll_${TS}i_${INTERVAL}.npy"
  "$DATASET_ROOT/dataset_train/pred_train_label_seqtoseq_alll_${TS}i_${INTERVAL}.npy"
  "$DATASET_ROOT/dataset_val/pred_val_img_seqtoseq_alll_${TS}i_${INTERVAL}.npy"
  "$DATASET_ROOT/dataset_val/pred_val_label_seqtoseq_alll_${TS}i_${INTERVAL}.npy"
)
if [[ "$GOES_VARIANT" != "none" ]]; then
  required+=(
    "$DATASET_ROOT/dataset_train/pred_train_${GOES_VARIANT}_seqtoseq_alll_${TS}i_${INTERVAL}.npy"
    "$DATASET_ROOT/dataset_val/pred_val_${GOES_VARIANT}_seqtoseq_alll_${TS}i_${INTERVAL}.npy"
  )
fi
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    printf 'Missing required dataset: %s\n' "$path" >&2
    exit 1
  fi
done

printf 'dataset=%s ts=%s models=%s seeds=%s goes=%s epochs=%s\n' \
  "$DATASET_ROOT" "$TS" "$MODELS" "$SEEDS" "$GOES_VARIANT" "$EPOCHS"

for model in $MODELS; do
  for seed in $SEEDS; do
    checkpoint_dir="$CHECKPOINT_BASE/${EXPERIMENT_TAG}_${GOES_VARIANT}_ts${TS}/${model}/seed${seed}"
    mkdir -p "$checkpoint_dir"
    export TS_SATFIRE_ORIG_CHECKPOINT_ROOT="$checkpoint_dir"

    command=(
      python legacy/original_ts_satfire/run_spatial_temp_model_pred_original.py
      -m "$model"
      -mode pred
      -r 0
      -nh 2
      -nc 43
      -ts "$TS"
      -it "$INTERVAL"
      -epochs "$EPOCHS"
      -seed "$seed"
      --training-profile paper
      --loss auto
      --checkpoint-metric "$CHECKPOINT_METRIC"
      --experiment-tag "$EXPERIMENT_TAG"
      --goes-variant "$GOES_VARIANT"
    )
    if [[ "$SAVE_PROBABILITY_MAPS" == "1" ]]; then
      command+=(--save-probability-maps)
    fi

    case "$model" in
      unet3d) batch_override="${UNET3D_BATCH:-}" ;;
      attunet) batch_override="${ATTUNET_BATCH:-}" ;;
      unetr3d) batch_override="${UNETR3D_BATCH:-}" ;;
      swinunetr3d) batch_override="${SWINUNETR3D_BATCH:-}" ;;
      *) batch_override="" ;;
    esac
    if [[ -n "$batch_override" ]]; then
      command+=(-b "$batch_override")
    fi

    printf '\n[%s seed=%s] ' "$model" "$seed"
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "$DRY_RUN" != "1" ]]; then
      "${command[@]}"
    fi
  done
done
