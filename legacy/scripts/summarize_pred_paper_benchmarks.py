from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PAIR_KEYS = ["model", "time_series_days", "interval", "seed"]
GROUP_KEYS = [
    "experiment_tag",
    "goes_variant",
    "model",
    "time_series_days",
    "training_profile",
    "loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paper-reproduction prediction benchmark results.")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--baseline-tag", default="viirs43")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_results(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("result_*.json")):
        with path.open() as handle:
            row = json.load(handle)
        row["result_path"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No result_*.json files found below {root}")
    return pd.DataFrame(rows)


def add_baseline_deltas(df: pd.DataFrame, baseline_tag: str) -> pd.DataFrame:
    baseline = df[df["experiment_tag"] == baseline_tag][PAIR_KEYS + ["test_f1", "test_iou"]].copy()
    baseline = baseline.rename(columns={"test_f1": "baseline_f1", "test_iou": "baseline_iou"})
    if baseline.duplicated(PAIR_KEYS).any():
        duplicates = baseline[baseline.duplicated(PAIR_KEYS, keep=False)]
        raise ValueError(f"Duplicate baseline rows for pairing keys:\n{duplicates}")
    merged = df.merge(baseline, on=PAIR_KEYS, how="left")
    merged["f1_delta_vs_baseline"] = merged["test_f1"] - merged["baseline_f1"]
    merged["iou_delta_vs_baseline"] = merged["test_iou"] - merged["baseline_iou"]
    return merged


def main() -> None:
    args = parse_args()
    detailed = add_baseline_deltas(load_results(args.checkpoint_root), args.baseline_tag)
    summary = (
        detailed.groupby(GROUP_KEYS, dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            f1_mean=("test_f1", "mean"),
            f1_std=("test_f1", "std"),
            iou_mean=("test_iou", "mean"),
            iou_std=("test_iou", "std"),
            f1_delta_mean=("f1_delta_vs_baseline", "mean"),
            iou_delta_mean=("iou_delta_vs_baseline", "mean"),
        )
        .reset_index()
        .sort_values(["time_series_days", "model", "experiment_tag"])
    )

    out = args.out or args.checkpoint_root / "pred_paper_benchmark_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    detailed.to_csv(out.with_name(f"{out.stem}_runs.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out}")
    print(f"Wrote {out.with_name(f'{out.stem}_runs.csv')}")


if __name__ == "__main__":
    main()

