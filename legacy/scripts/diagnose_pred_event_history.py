from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from eval_pred_event_candidate_masks import (
    DEFAULT_CANDIDATE_ROOT,
    GEOMETRY_CAT,
    GEOMETRY_NUM,
    GOES_FRP,
    TARGET,
    build_model,
    candidate_path,
    goes_history_features,
    history_num_features,
    load_split,
)

ID_COLS = ["fire_id", "date", "day_idx", "component_id", TARGET]


def parse_history_days(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(x < 1 for x in values):
        raise ValueError("--history-days must contain positive integers, e.g. 2 or 2,4")
    return values


def split_path(root: Path, split: str, args: argparse.Namespace, history_days: int) -> Path:
    return candidate_path(
        root,
        split,
        args.connectivity,
        args.candidate_radius,
        args.min_component_pixels,
        history_days,
    )


def overview(path: Path, chunksize: int) -> tuple[dict, pd.DataFrame]:
    rows = 0
    positives = 0
    fire_ids: set[str] = set()
    fire_dates: set[tuple[str, str]] = set()
    components: set[tuple[str, str, int]] = set()
    by_day: dict[int, dict[str, int]] = {}

    for chunk in pd.read_csv(path, usecols=ID_COLS, chunksize=chunksize):
        rows += len(chunk)
        positives += int(chunk[TARGET].sum())
        fire_ids.update(chunk["fire_id"].astype(str).unique())
        fire_dates.update(zip(chunk["fire_id"].astype(str), chunk["date"].astype(str)))
        components.update(
            zip(
                chunk["fire_id"].astype(str),
                chunk["date"].astype(str),
                chunk["component_id"].astype(int),
            )
        )
        day_g = chunk.groupby("day_idx")[TARGET].agg(["size", "sum"])
        for day_idx, row in day_g.iterrows():
            item = by_day.setdefault(int(day_idx), {"rows": 0, "positives": 0})
            item["rows"] += int(row["size"])
            item["positives"] += int(row["sum"])

    summary = {
        "rows": rows,
        "positives": positives,
        "positive_rate": positives / rows if rows else np.nan,
        "n_fires": len(fire_ids),
        "n_fire_dates": len(fire_dates),
        "n_components": len(components),
        "min_day_idx": min(by_day) if by_day else np.nan,
        "max_day_idx": max(by_day) if by_day else np.nan,
    }
    by_day_rows = []
    for day_idx, item in sorted(by_day.items()):
        by_day_rows.append({
            "day_idx": day_idx,
            "rows": item["rows"],
            "positives": item["positives"],
            "positive_rate": item["positives"] / item["rows"] if item["rows"] else np.nan,
        })
    return summary, pd.DataFrame(by_day_rows)


def label_feature_means(path: Path, features: list[str], chunksize: int) -> pd.DataFrame:
    features = list(dict.fromkeys(features))
    sums = {0: np.zeros(len(features), dtype=np.float64), 1: np.zeros(len(features), dtype=np.float64)}
    counts = {0: 0, 1: 0}

    for chunk in pd.read_csv(path, usecols=[TARGET] + features, chunksize=chunksize):
        for label in [0, 1]:
            g = chunk[chunk[TARGET] == label]
            if g.empty:
                continue
            counts[label] += len(g)
            sums[label] += g[features].fillna(0).to_numpy(dtype=np.float64).sum(axis=0)

    rows = []
    for idx, feature in enumerate(features):
        mean0 = sums[0][idx] / counts[0] if counts[0] else np.nan
        mean1 = sums[1][idx] / counts[1] if counts[1] else np.nan
        rows.append({
            "feature": feature,
            "negative_mean": mean0,
            "positive_mean": mean1,
            "positive_minus_negative": mean1 - mean0,
            "positive_over_negative": mean1 / mean0 if mean0 not in (0, np.nan) and mean0 != 0 else np.nan,
        })
    return pd.DataFrame(rows)


def auc_ablation(root: Path, args: argparse.Namespace, history_days: int) -> pd.DataFrame:
    geom = GEOMETRY_NUM
    hist = history_num_features(history_days)
    goes_current = GOES_FRP
    goes_hist = goes_history_features(history_days)
    variants = [
        ("geometry_no_history", geom, GEOMETRY_CAT),
        ("geometry_plus_viirs_history", geom + hist, GEOMETRY_CAT),
        ("geometry_viirs_history_plus_goes_current", geom + hist + goes_current, GEOMETRY_CAT),
        ("full_viirs_goes_history", geom + hist + goes_current + goes_hist, GEOMETRY_CAT),
    ]

    train_path = split_path(root, "train", args, history_days)
    val_path = split_path(root, "val", args, history_days)
    test_path = split_path(root, "test", args, history_days)

    rows = []
    for name, num_features, cat_features in variants:
        features = num_features + cat_features
        X_train, y_train, _ = load_split(train_path, features, sample=args.train_sample)
        X_val, y_val, _ = load_split(val_path, features)
        X_test, y_test, _ = load_split(test_path, features)
        model = build_model(num_features, cat_features)
        model.fit(X_train, y_train)
        for split, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            prob = model.predict_proba(X)[:, 1]
            rows.append({
                "history_days": history_days,
                "variant": name,
                "split": split,
                "pr_auc": average_precision_score(y, prob),
                "roc_auc": roc_auc_score(y, prob),
                "n_features": len(features),
            })
        print(f"finished ablation history_days={history_days} variant={name}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose how event candidate history windows are used.")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT)
    parser.add_argument("--history-days", default="2,4", help="Comma-separated history windows to inspect, e.g. 2,4")
    parser.add_argument("--connectivity", type=int, default=8)
    parser.add_argument("--candidate-radius", type=float, default=5.0)
    parser.add_argument("--min-component-pixels", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--train-sample", type=int, default=1_000_000)
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    root = args.candidate_root
    history_values = parse_history_days(args.history_days)
    overview_rows = []
    by_day_outputs = []
    label_mean_outputs = []
    ablation_outputs = []

    for h in history_values:
        feature_cols = history_num_features(h) + GOES_FRP + goes_history_features(h)
        for split in ["train", "val", "test"]:
            path = split_path(root, split, args, h)
            if not path.exists():
                raise FileNotFoundError(path)
            summary, by_day = overview(path, args.chunksize)
            summary.update({"history_days": h, "split": split, "path": str(path)})
            overview_rows.append(summary)
            by_day.insert(0, "split", split)
            by_day.insert(0, "history_days", h)
            by_day_outputs.append(by_day)
            means = label_feature_means(path, feature_cols, args.chunksize)
            means.insert(0, "split", split)
            means.insert(0, "history_days", h)
            label_mean_outputs.append(means)
            print(f"history_days={h} split={split} rows={summary['rows']} positives={summary['positives']} positive_rate={summary['positive_rate']:.6f} day_idx={summary['min_day_idx']}..{summary['max_day_idx']}")

        if not args.skip_ablation:
            ablation_outputs.append(auc_ablation(root, args, h))

    overview_df = pd.DataFrame(overview_rows)
    by_day_df = pd.concat(by_day_outputs, ignore_index=True)
    means_df = pd.concat(label_mean_outputs, ignore_index=True)

    overview_out = root / "pred_event_history_diagnostics_overview.csv"
    by_day_out = root / "pred_event_history_diagnostics_by_day_idx.csv"
    means_out = root / "pred_event_history_diagnostics_label_means.csv"
    overview_df.to_csv(overview_out, index=False)
    by_day_df.to_csv(by_day_out, index=False)
    means_df.to_csv(means_out, index=False)
    print("Wrote", overview_out)
    print("Wrote", by_day_out)
    print("Wrote", means_out)

    if ablation_outputs:
        ablation_df = pd.concat(ablation_outputs, ignore_index=True)
        ablation_out = root / "pred_event_history_diagnostics_ablation_auc.csv"
        ablation_df.to_csv(ablation_out, index=False)
        print("Wrote", ablation_out)
        print("\nAblation:")
        print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
