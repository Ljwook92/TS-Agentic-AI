import argparse
import json
import os
import re
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from calibration import class_conditional_quantiles, evaluate_cp
from satimg_dataset_processor.data_generator_pred_torch import FireDataset as PredFireDataset
from satimg_dataset_processor.data_generator_torch import FireDataset as AfBaFireDataset
from satimg_dataset_processor.data_generator_torch import Normalize
from spatial_models.swinunetr.swinunetr import SwinUNETR

ROOT_DIR = os.path.expanduser("~/data/SatFire")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AF_TEST_IDS = [
    "elephant_hill_fire",
    "eagle_bluff_fire",
    "double_creek_fire",
    "sparks_lake_fire",
    "lytton_fire",
    "chuckegg_creek_fire",
    "swedish_fire",
    "sydney_fire",
    "thomas_fire",
    "tubbs_fire",
    "carr_fire",
    "camp_fire",
    "creek_fire",
    "blue_ridge_fire",
    "dixie_fire",
    "mosquito_fire",
    "calfcanyon_fire",
]


class EmptyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset")


def build_swinunetr3d(ts_length: int, n_channel: int, num_heads: int, hidden_size: int, attn_version: str) -> nn.Module:
    image_size = (ts_length, 256, 256)
    patch_size = (1, 2, 2)
    window_size = (ts_length, 4, 4)
    model = SwinUNETR(
        image_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        in_channels=n_channel,
        out_channels=2,
        depths=(2, 2, 2, 2),
        num_heads=(num_heads, num_heads, num_heads, num_heads),
        feature_size=hidden_size,
        norm_name="batch",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        attn_version=attn_version,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
    )
    return nn.DataParallel(model)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def _extract_epoch(path: str) -> int:
    m = re.search(r"checkpoint_epoch_(\d+)", os.path.basename(path))
    if not m:
        return -1
    return int(m.group(1))


def resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if args.ckpt:
        return args.ckpt

    ckpt_dir = args.ckpt_dir
    pattern = (
        f"model_swinunetr3d_run_{args.run}_seed_{args.seed}_mode_{args.mode}_"
        f"num_heads_{args.nh}_hidden_size_{args.ed}_batchsize_*_"
        f"checkpoint_epoch_*_nc_{args.nc}_ts_{args.ts}_attention_{args.av}.pth"
    )
    candidates = sorted(glob(os.path.join(ckpt_dir, pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matched pattern under {ckpt_dir}: {pattern}"
        )

    if args.epoch is not None:
        exact = [p for p in candidates if _extract_epoch(p) == args.epoch]
        if not exact:
            raise FileNotFoundError(
                f"No checkpoint matched requested epoch={args.epoch} under {ckpt_dir}"
            )
        return exact[-1]

    return max(candidates, key=_extract_epoch)


def get_fire_prob_and_label(
    batch: Dict[str, torch.Tensor], logits: torch.Tensor, mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_fire = torch.sigmoid(logits[:, 1, ...])
    raw_labels = batch["labels"][:, 1, ...].to(probs_fire.device)

    valid_mask = raw_labels != -1
    y = (raw_labels > 0).long()

    return (
        probs_fire.detach().cpu().numpy(),
        y.detach().cpu().numpy().astype(np.int64),
        valid_mask.detach().cpu().numpy().astype(bool),
    )


def collect_prob_and_label(model: nn.Module, dataloader: DataLoader, device: torch.device, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    all_prob: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x = batch["data"].to(device)
            logits = model(x)
            if mode == "pred":
                logits = logits.mean(2)
            prob_fire, y_true, valid = get_fire_prob_and_label(batch, logits, mode)
            prob_flat = prob_fire.reshape(-1)
            y_flat = y_true.reshape(-1)
            valid_flat = valid.reshape(-1)
            all_prob.append(prob_flat[valid_flat])
            all_y.append(y_flat[valid_flat])

    if not all_prob:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    return np.concatenate(all_prob), np.concatenate(all_y)


def build_afba_calibration_dataset(mode: str, ts_length: int, interval: int, n_channel: int) -> torch.utils.data.Dataset:
    if mode == "af":
        transform = Normalize(
            mean=[18.76488, 27.441864, 20.584806, 305.99478, 294.31738, 14.625097, 276.4207, 275.16766],
            std=[15.911591, 14.879259, 10.832616, 21.761852, 24.703484, 9.878246, 40.64329, 40.7657],
        )
        label_sel = 2
    else:
        transform = Normalize(
            mean=[17.952442, 26.94709, 19.82838, 317.80234, 308.47693, 13.87255, 291.0257, 288.9398],
            std=[15.359564, 14.336508, 10.64194, 12.505946, 11.571564, 9.666024, 11.495529, 7.9788895],
        )
        label_sel = 0

    image_path = os.path.join(DATASET_DIR, "dataset_val", f"{mode}_val_img_seqtoseq_alll_{ts_length}i_{interval}.npy")
    label_path = os.path.join(DATASET_DIR, "dataset_val", f"{mode}_val_label_seqtoseq_alll_{ts_length}i_{interval}.npy")
    return AfBaFireDataset(
        image_path=image_path,
        label_path=label_path,
        ts_length=ts_length,
        transform=transform,
        n_channel=n_channel,
        label_sel=label_sel,
    )


def build_af_test_cases(ts_length: int, interval: int, n_channel: int) -> List[Tuple[str, torch.utils.data.Dataset]]:
    cases: List[Tuple[str, torch.utils.data.Dataset]] = []
    transform = Normalize(
        mean=[18.76488, 27.441864, 20.584806, 305.99478, 294.31738, 14.625097, 276.4207, 275.16766],
        std=[15.911591, 14.879259, 10.832616, 21.761852, 24.703484, 9.878246, 40.64329, 40.7657],
    )

    for fire_id in AF_TEST_IDS:
        image_path = os.path.join(DATASET_DIR, f"af_{fire_id}_img_seqtoseql_{ts_length}i_{interval}.npy")
        label_path = os.path.join(DATASET_DIR, f"af_{fire_id}_label_seqtoseql_{ts_length}i_{interval}.npy")
        if os.path.exists(image_path) and os.path.exists(label_path):
            ds = AfBaFireDataset(
                image_path=image_path,
                label_path=label_path,
                ts_length=ts_length,
                transform=transform,
                n_channel=n_channel,
                label_sel=2,
            )
            cases.append((fire_id, ds))

    return cases


def build_ba_test_cases(ts_length: int, interval: int, n_channel: int) -> List[Tuple[str, torch.utils.data.Dataset]]:
    df = pd.read_csv(os.path.join(BASE_DIR, "roi", "us_fire_2021_out_new.csv"))
    ids = df["Id"].astype(str).tolist()
    label_sels = df["label_sel"].astype(int).tolist()

    transform = Normalize(
        mean=[17.952442, 26.94709, 19.82838, 317.80234, 308.47693, 13.87255, 291.0257, 288.9398],
        std=[15.359564, 14.336508, 10.64194, 12.505946, 11.571564, 9.666024, 11.495529, 7.9788895],
    )

    cases: List[Tuple[str, torch.utils.data.Dataset]] = []
    for fire_id, label_sel in zip(ids, label_sels):
        image_path = os.path.join(
            DATASET_DIR,
            "dataset_test",
            f"ba_{fire_id}_img_seqtoseql_{ts_length}i_{interval}.npy",
        )
        label_path = os.path.join(
            DATASET_DIR,
            "dataset_test",
            f"ba_{fire_id}_label_seqtoseql_{ts_length}i_{interval}.npy",
        )
        if os.path.exists(image_path) and os.path.exists(label_path):
            ds = AfBaFireDataset(
                image_path=image_path,
                label_path=label_path,
                ts_length=ts_length,
                transform=transform,
                n_channel=n_channel,
                label_sel=label_sel,
            )
            cases.append((fire_id, ds))

    return cases


def build_pred_calibration_dataset(ts_length: int, interval: int, n_channel: int) -> torch.utils.data.Dataset:
    image_path = os.path.join(DATASET_DIR, "dataset_val", f"pred_val_img_seqtoseq_alll_{ts_length}i_{interval}.npy")
    label_path = os.path.join(DATASET_DIR, "dataset_val", f"pred_val_label_seqtoseq_alll_{ts_length}i_{interval}.npy")
    return PredFireDataset(
        image_path=image_path,
        label_path=label_path,
        ts_length=ts_length,
        n_channel=n_channel,
        label_sel=0,
        target_is_single_day=True,
        use_augmentations=False,
    )


def build_pred_test_cases(ts_length: int, interval: int, n_channel: int) -> List[Tuple[str, torch.utils.data.Dataset]]:
    df = pd.read_csv(os.path.join(BASE_DIR, "roi", "us_fire_2021_out_new.csv"))
    ids = [x for x in df["Id"].astype(str).tolist() if x != "US_2021_NV3700011641620210517"]

    cases: List[Tuple[str, torch.utils.data.Dataset]] = []
    for fire_id in ids:
        image_path = os.path.join(DATASET_DIR, "dataset_test", f"pred_{fire_id}_img_seqtoseql_{ts_length}i_{interval}.npy")
        label_path = os.path.join(DATASET_DIR, "dataset_test", f"pred_{fire_id}_label_seqtoseql_{ts_length}i_{interval}.npy")
        if os.path.exists(image_path) and os.path.exists(label_path):
            ds = PredFireDataset(
                image_path=image_path,
                label_path=label_path,
                ts_length=ts_length,
                n_channel=n_channel,
                label_sel=0,
                target_is_single_day=True,
                use_augmentations=False,
            )
            cases.append((fire_id, ds))

    return cases


def build_datasets(
    mode: str, ts_length: int, interval: int, n_channel: int
) -> Tuple[torch.utils.data.Dataset, List[Tuple[str, torch.utils.data.Dataset]]]:
    if mode == "af":
        cal_ds = build_afba_calibration_dataset(mode="af", ts_length=ts_length, interval=interval, n_channel=n_channel)
        test_cases = build_af_test_cases(ts_length=ts_length, interval=interval, n_channel=n_channel)
    elif mode == "ba":
        cal_ds = build_afba_calibration_dataset(mode="ba", ts_length=ts_length, interval=interval, n_channel=n_channel)
        test_cases = build_ba_test_cases(ts_length=ts_length, interval=interval, n_channel=n_channel)
    elif mode == "pred":
        cal_ds = build_pred_calibration_dataset(ts_length=ts_length, interval=interval, n_channel=n_channel)
        test_cases = build_pred_test_cases(ts_length=ts_length, interval=interval, n_channel=n_channel)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return cal_ds, test_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Conformal prediction analysis for TS-SatFire (SwinUNETR-3D)")
    parser.add_argument("-mode", type=str, choices=["af", "ba", "pred"], required=True)
    parser.add_argument("-ckpt", type=str, default=None, help="Path to trained SwinUNETR-3D checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default=os.path.join(ROOT_DIR, "checkpoints"))
    parser.add_argument("-epoch", type=int, default=None, help="Epoch to load when -ckpt is omitted")
    parser.add_argument("-run", type=int, default=1, help="Training run id used in checkpoint filename")
    parser.add_argument("-seed", type=int, default=42, help="Training seed used in checkpoint filename")
    parser.add_argument("-b", type=int, default=2, help="Batch size")
    parser.add_argument("-nh", type=int, required=True, help="SwinUNETR num_heads")
    parser.add_argument("-ed", type=int, required=True, help="SwinUNETR feature size")
    parser.add_argument("-nc", type=int, required=True, help="Number of channels")
    parser.add_argument("-ts", type=int, required=True, help="Time-series length")
    parser.add_argument("-it", type=int, required=True, help="Interval")
    parser.add_argument("-av", type=str, default="v1", help="Attention version for SwinUNETR")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional output JSON path (default: cp_metrics_<mode>.json in project root)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_swinunetr3d(
        ts_length=args.ts,
        n_channel=args.nc,
        num_heads=args.nh,
        hidden_size=args.ed,
        attn_version=args.av,
    ).to(device)

    ckpt_path = resolve_checkpoint_path(args)
    load_checkpoint(model, ckpt_path, device)

    cal_ds, test_cases = build_datasets(
        mode=args.mode,
        ts_length=args.ts,
        interval=args.it,
        n_channel=args.nc,
    )

    if len(cal_ds) == 0:
        raise RuntimeError("Calibration dataset is empty.")
    if len(test_cases) == 0:
        raise RuntimeError("Test dataset is empty. Check dataset generation and file naming.")

    cal_loader = DataLoader(cal_ds, batch_size=args.b, shuffle=False)
    cal_prob, cal_y = collect_prob_and_label(model, cal_loader, device, args.mode)
    qhat = class_conditional_quantiles(cal_prob, cal_y, alpha=args.alpha)

    case_metrics: List[Dict[str, float]] = []
    test_probs: List[np.ndarray] = []
    test_labels: List[np.ndarray] = []
    for case_id, case_ds in test_cases:
        case_loader = DataLoader(case_ds, batch_size=args.b, shuffle=False)
        case_prob, case_y = collect_prob_and_label(model, case_loader, device, args.mode)
        test_probs.append(case_prob)
        test_labels.append(case_y)
        m = evaluate_cp(case_prob, case_y, qhat=qhat, threshold=args.threshold)
        m["case_id"] = case_id
        case_metrics.append(m)

    test_prob = np.concatenate(test_probs) if test_probs else np.array([], dtype=np.float64)
    test_y = np.concatenate(test_labels) if test_labels else np.array([], dtype=np.int64)
    global_metrics = evaluate_cp(test_prob, test_y, qhat=qhat, threshold=args.threshold)

    macro_keys = [
        "coverage",
        "uncertainty_rate",
        "error_capture_rate",
        "baseline_f1",
        "baseline_iou",
        "selective_f1",
        "selective_iou",
    ]
    macro_metrics: Dict[str, float] = {}
    for key in macro_keys:
        vals = np.array([cm[key] for cm in case_metrics], dtype=np.float64)
        macro_metrics[key] = float(np.nanmean(vals))

    result = {
        "mode": args.mode,
        "checkpoint": ckpt_path,
        "alpha": args.alpha,
        "threshold": args.threshold,
        "calibration_pixels": int(cal_y.size),
        "test_pixels": int(test_y.size),
        "metrics_global_pixel": global_metrics,
        "metrics_macro_case_mean": macro_metrics,
        "num_test_cases": len(case_metrics),
        "case_metrics": case_metrics,
    }

    out_json = args.out_json or os.path.join(BASE_DIR, f"cp_metrics_{args.mode}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved CP metrics to: {out_json}")


if __name__ == "__main__":
    main()
