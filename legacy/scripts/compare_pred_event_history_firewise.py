from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_ROOT = Path("/home/jlc3q/data/SatFire/event_candidates")


def threshold_tag(value: float) -> str:
    return str(value).replace(".", "p")


def resolve_firewise_path(root: Path, history_days: int, variant: str, threshold: float) -> Path:
    stem = f"pred_event_mask_eval_firewise_h{history_days}_partial_{variant}"
    candidates = [
        root / f"{stem}_thr{threshold_tag(threshold)}.csv",
        root / f"{stem}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No firewise result found. Tried: {candidates}")


def load_test_rows(path: Path, model: str, threshold: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "split", "fire_id", "selected_method", "selected_value", "tp", "fp", "fn", "iou", "f1"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df[(df["split"] == "test") & (df["model"] == model)].copy()
    if df.empty:
        raise ValueError(f"No test rows for model={model} in {path}")
    if df["fire_id"].duplicated().any():
        raise ValueError(f"Duplicate test fire_id rows in {path}")
    if not (df["selected_method"] == "threshold").all():
        raise ValueError(f"Non-threshold result found in {path}")
    if not np.allclose(df["selected_value"].to_numpy(dtype=float), threshold):
        values = sorted(df["selected_value"].unique())
        raise ValueError(f"Expected threshold={threshold}, found {values} in {path}")
    return df


def confusion_metric(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, metric: str) -> np.ndarray:
    if metric == "iou":
        numerator = tp
        denominator = tp + fp + fn
    elif metric == "f1":
        numerator = 2 * tp
        denominator = 2 * tp + fp + fn
    else:
        raise ValueError(metric)
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def paired_summary(
    merged: pd.DataFrame,
    metric: str,
    sample_indices: np.ndarray,
) -> list[dict]:
    a = merged[f"{metric}_a"].to_numpy(dtype=np.float64)
    b = merged[f"{metric}_b"].to_numpy(dtype=np.float64)
    delta = b - a
    tied = np.isclose(delta, 0.0)
    bootstrap_macro = delta[sample_indices].mean(axis=1)
    try:
        wilcoxon_p = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        wilcoxon_p = 1.0

    rows = [{
        "aggregation": "fire_macro",
        "metric": metric,
        "history_a": float(a.mean()),
        "history_b": float(b.mean()),
        "delta_b_minus_a": float(delta.mean()),
        "median_fire_delta": float(np.median(delta)),
        "ci95_low": float(np.percentile(bootstrap_macro, 2.5)),
        "ci95_high": float(np.percentile(bootstrap_macro, 97.5)),
        "bootstrap_probability_positive": float(np.mean(bootstrap_macro > 0)),
        "wilcoxon_p": wilcoxon_p,
        "fires_better": int(np.sum((delta > 0) & ~tied)),
        "fires_tied": int(np.sum(tied)),
        "fires_worse": int(np.sum((delta < 0) & ~tied)),
    }]

    boot_metrics = {}
    observed = {}
    for suffix in ["a", "b"]:
        tp = merged[f"tp_{suffix}"].to_numpy(dtype=np.float64)
        fp = merged[f"fp_{suffix}"].to_numpy(dtype=np.float64)
        fn = merged[f"fn_{suffix}"].to_numpy(dtype=np.float64)
        observed[suffix] = float(confusion_metric(tp.sum(keepdims=True), fp.sum(keepdims=True), fn.sum(keepdims=True), metric)[0])
        boot_metrics[suffix] = confusion_metric(
            tp[sample_indices].sum(axis=1),
            fp[sample_indices].sum(axis=1),
            fn[sample_indices].sum(axis=1),
            metric,
        )
    bootstrap_micro_delta = boot_metrics["b"] - boot_metrics["a"]
    rows.append({
        "aggregation": "fire_bootstrap_micro",
        "metric": metric,
        "history_a": observed["a"],
        "history_b": observed["b"],
        "delta_b_minus_a": observed["b"] - observed["a"],
        "median_fire_delta": np.nan,
        "ci95_low": float(np.percentile(bootstrap_micro_delta, 2.5)),
        "ci95_high": float(np.percentile(bootstrap_micro_delta, 97.5)),
        "bootstrap_probability_positive": float(np.mean(bootstrap_micro_delta > 0)),
        "wilcoxon_p": np.nan,
        "fires_better": np.nan,
        "fires_tied": np.nan,
        "fires_worse": np.nan,
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired fire-level bootstrap comparison of candidate history windows.")
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--history-a", type=int, default=4)
    parser.add_argument("--history-b", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--variant", default="goes_frp_motion_firepred")
    parser.add_argument("--model", default="geometry_plus_goes_frp_motion_firepred")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be >= 1")
    path_a = resolve_firewise_path(args.candidate_root, args.history_a, args.variant, args.threshold)
    path_b = resolve_firewise_path(args.candidate_root, args.history_b, args.variant, args.threshold)
    a = load_test_rows(path_a, args.model, args.threshold)
    b = load_test_rows(path_b, args.model, args.threshold)
    merged = a.merge(b, on="fire_id", suffixes=("_a", "_b"), how="inner", validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        only_a = sorted(set(a["fire_id"]) - set(b["fire_id"]))
        only_b = sorted(set(b["fire_id"]) - set(a["fire_id"]))
        raise ValueError(f"Fire IDs do not match. only_a={only_a}, only_b={only_b}")

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(0, len(merged), size=(args.bootstrap_samples, len(merged)))
    rows = []
    for metric in ["iou", "f1"]:
        rows.extend(paired_summary(merged, metric, sample_indices))
    result = pd.DataFrame(rows)
    result.insert(0, "n_fires", len(merged))
    result.insert(0, "history_b_days", args.history_b)
    result.insert(0, "history_a_days", args.history_a)

    out = args.candidate_root / (
        f"pred_event_history_comparison_h{args.history_a}_vs_h{args.history_b}_"
        f"thr{threshold_tag(args.threshold)}.csv"
    )
    result.to_csv(out, index=False)
    print(f"history_a={args.history_a} path={path_a}")
    print(f"history_b={args.history_b} path={path_b}")
    print(f"paired_fires={len(merged)} bootstrap_samples={args.bootstrap_samples}")
    print(result.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
