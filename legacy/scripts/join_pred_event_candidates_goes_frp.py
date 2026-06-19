from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from dataset_gen_pred_goes_spatial import (
    DEFAULT_GOES_ROOT,
    collect_goes_files_by_day,
    crop_profile,
    daily_maps,
    find_event_dir,
    firepred_path_from_viirs,
    viirs_day_files,
)


DEFAULT_CANDIDATE_ROOT = "/home/jlc3q/data/SatFire/event_candidates"


def read_in_chunks(path: Path, chunksize: int):
    return pd.read_csv(path, chunksize=chunksize)


def radius_tag(value: float) -> str:
    return str(value).replace(".", "p")


def history_suffix(history_days: int, allow_partial_history: bool = False) -> str:
    if history_days <= 1:
        return ""
    partial_tag = "_partial" if allow_partial_history else ""
    return f"_h{history_days}{partial_tag}"


def date_lag(date: str, lag: int) -> str:
    return (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=lag)).strftime("%Y-%m-%d")


def local_window_stats(arr: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    size = 2 * radius + 1
    max_map = ndimage.maximum_filter(arr, size=size, mode="constant", cval=0.0)
    mean_map = ndimage.uniform_filter(arr.astype(np.float32), size=size, mode="constant", cval=0.0)
    return max_map.astype(np.float32), mean_map.astype(np.float32)


def weighted_centroid(weight: np.ndarray) -> tuple[float, float, float]:
    total = float(np.nansum(weight))
    if total <= 0:
        return np.nan, np.nan, 0.0
    rows, cols = np.indices(weight.shape)
    centroid_row = float(np.nansum(rows * weight) / total)
    centroid_col = float(np.nansum(cols * weight) / total)
    return centroid_row, centroid_col, total


def cosine_alignment(row_a: np.ndarray, col_a: np.ndarray, row_b: np.ndarray, col_b: np.ndarray) -> np.ndarray:
    dot = row_a * row_b + col_a * col_b
    norm_a = np.sqrt(row_a * row_a + col_a * col_a)
    norm_b = np.sqrt(row_b * row_b + col_b * col_b)
    denom = norm_a * norm_b
    out = np.zeros_like(dot, dtype=np.float32)
    valid = denom > 0
    out[valid] = (dot[valid] / denom[valid]).astype(np.float32)
    return out


class GoesFrpCache:
    def __init__(self, goes_root: Path):
        self.goes_root = goes_root
        self.event_cache: dict[str, dict[str, dict[str, list[Path]]]] = {}
        self.profile_cache: dict[str, dict] = {}
        self.day_cache: dict[tuple[str, str], dict[str, np.ndarray | float]] = {}

    def get_profile(self, fire_id: str) -> dict | None:
        if fire_id in self.profile_cache:
            return self.profile_cache[fire_id]
        files = viirs_day_files(fire_id)
        if not files:
            return None
        ref_firepred = firepred_path_from_viirs(files[0])
        if not ref_firepred.exists():
            return None
        profile = crop_profile(ref_firepred)
        self.profile_cache[fire_id] = profile
        return profile

    def get_goes_by_day(self, fire_id: str) -> dict[str, dict[str, list[Path]]]:
        if fire_id in self.event_cache:
            return self.event_cache[fire_id]
        event_dir = find_event_dir(self.goes_root, fire_id)
        if event_dir is None:
            self.event_cache[fire_id] = {}
            return {}
        goes_by_day = collect_goes_files_by_day(event_dir)
        self.event_cache[fire_id] = goes_by_day
        return goes_by_day

    def get_day_features(self, fire_id: str, date: str, high_frp_percentile: float) -> dict[str, np.ndarray | float]:
        key = (fire_id, date)
        if key in self.day_cache:
            return self.day_cache[key]

        profile = self.get_profile(fire_id)
        if profile is None:
            zeros = np.zeros((256, 256), dtype=np.float32)
            features = self._empty_day_features(zeros)
            self.day_cache[key] = features
            return features

        goes_by_day = self.get_goes_by_day(fire_id)
        day = daily_maps(date, goes_by_day, profile)
        frp_sum = np.asarray(day["frp_sum_log1p"], dtype=np.float32)
        frp_max = np.asarray(day["frp_max_log1p"], dtype=np.float32)
        active = np.asarray(day["daily_active"], dtype=np.float32)
        active_frequency = np.asarray(day["active_frequency"], dtype=np.float32)

        # Weighted active emphasizes active pixels with strong FRP and suppresses weak/no-FRP active detections.
        positive_frp = frp_sum[frp_sum > 0]
        p95 = float(np.nanpercentile(positive_frp, 95)) if positive_frp.size else 0.0
        if p95 > 0:
            weighted_active = active * np.clip(frp_sum / p95, 0.0, 1.0)
        else:
            weighted_active = np.zeros_like(frp_sum, dtype=np.float32)

        high_frp = np.zeros_like(frp_sum, dtype=bool)
        if positive_frp.size:
            threshold = float(np.nanpercentile(positive_frp, high_frp_percentile))
            high_frp = frp_sum >= threshold

        if np.any(high_frp):
            distance_to_high_frp = ndimage.distance_transform_edt(~high_frp).astype(np.float32)
        else:
            distance_to_high_frp = np.full_like(frp_sum, np.nan, dtype=np.float32)

        frp_centroid_row, frp_centroid_col, frp_total = weighted_centroid(frp_sum)
        weighted_centroid_row, weighted_centroid_col, weighted_total = weighted_centroid(weighted_active)

        features = {
            "frp_sum": frp_sum,
            "frp_max": frp_max,
            "active": active,
            "active_frequency": active_frequency,
            "weighted_active": weighted_active.astype(np.float32),
            "distance_to_high_frp": distance_to_high_frp,
            "frp_sum_5x5_max": local_window_stats(frp_sum, radius=2)[0],
            "frp_sum_5x5_mean": local_window_stats(frp_sum, radius=2)[1],
            "frp_max_5x5_max": local_window_stats(frp_max, radius=2)[0],
            "weighted_active_5x5_max": local_window_stats(weighted_active.astype(np.float32), radius=2)[0],
            "weighted_active_5x5_mean": local_window_stats(weighted_active.astype(np.float32), radius=2)[1],
            "goes_frp_centroid_row": frp_centroid_row,
            "goes_frp_centroid_col": frp_centroid_col,
            "goes_frp_total_log1p": frp_total,
            "goes_weighted_active_centroid_row": weighted_centroid_row,
            "goes_weighted_active_centroid_col": weighted_centroid_col,
            "goes_weighted_active_total": weighted_total,
        }
        self.day_cache[key] = features
        return features

    def empty_day_features(self) -> dict[str, np.ndarray | float]:
        return self._empty_day_features(np.zeros((256, 256), dtype=np.float32))

    @staticmethod
    def _empty_day_features(base: np.ndarray) -> dict[str, np.ndarray | float]:
        nan_map = np.full_like(base, np.nan, dtype=np.float32)
        return {
            "frp_sum": base,
            "frp_max": base,
            "active": base,
            "active_frequency": base,
            "weighted_active": base,
            "distance_to_high_frp": nan_map,
            "frp_sum_5x5_max": base,
            "frp_sum_5x5_mean": base,
            "frp_max_5x5_max": base,
            "weighted_active_5x5_max": base,
            "weighted_active_5x5_mean": base,
            "goes_frp_centroid_row": np.nan,
            "goes_frp_centroid_col": np.nan,
            "goes_frp_total_log1p": 0.0,
            "goes_weighted_active_centroid_row": np.nan,
            "goes_weighted_active_centroid_col": np.nan,
            "goes_weighted_active_total": 0.0,
        }


def add_goes_features(
    group: pd.DataFrame,
    day_features: dict[str, np.ndarray | float],
    history_day_features: list[dict[str, np.ndarray | float]] | None = None,
    available_goes_history_days: int | None = None,
) -> pd.DataFrame:
    out = group.copy()
    rr = out["candidate_row"].to_numpy(dtype=np.int64)
    cc = out["candidate_col"].to_numpy(dtype=np.int64)
    if available_goes_history_days is None:
        available_goes_history_days = len(history_day_features) if history_day_features else 1
    out["available_goes_history_days"] = int(available_goes_history_days)
    out["goes_history_coverage_fraction"] = (
        float(available_goes_history_days) / float(len(history_day_features))
        if history_day_features else 1.0
    )

    for name, source_key in [
        ("goes_frp_sum_at_candidate", "frp_sum"),
        ("goes_frp_max_at_candidate", "frp_max"),
        ("goes_active_at_candidate", "active"),
        ("goes_active_frequency_at_candidate", "active_frequency"),
        ("goes_weighted_active_at_candidate", "weighted_active"),
        ("goes_distance_to_high_frp", "distance_to_high_frp"),
        ("goes_frp_sum_5x5_max", "frp_sum_5x5_max"),
        ("goes_frp_sum_5x5_mean", "frp_sum_5x5_mean"),
        ("goes_frp_max_5x5_max", "frp_max_5x5_max"),
        ("goes_weighted_active_5x5_max", "weighted_active_5x5_max"),
        ("goes_weighted_active_5x5_mean", "weighted_active_5x5_mean"),
    ]:
        arr = np.asarray(day_features[source_key])
        out[name] = arr[rr, cc].astype(np.float32)

    for name in [
        "goes_frp_centroid_row",
        "goes_frp_centroid_col",
        "goes_frp_total_log1p",
        "goes_weighted_active_centroid_row",
        "goes_weighted_active_centroid_col",
        "goes_weighted_active_total",
    ]:
        out[name] = day_features[name]

    cand_row_delta = out["candidate_row"].to_numpy(dtype=np.float32) - out["nearest_fire_row"].to_numpy(dtype=np.float32)
    cand_col_delta = out["candidate_col"].to_numpy(dtype=np.float32) - out["nearest_fire_col"].to_numpy(dtype=np.float32)

    frp_row_delta = float(day_features["goes_frp_centroid_row"]) - out["nearest_fire_row"].to_numpy(dtype=np.float32)
    frp_col_delta = float(day_features["goes_frp_centroid_col"]) - out["nearest_fire_col"].to_numpy(dtype=np.float32)
    out["goes_candidate_frp_centroid_alignment"] = cosine_alignment(
        cand_row_delta,
        cand_col_delta,
        frp_row_delta.astype(np.float32),
        frp_col_delta.astype(np.float32),
    )

    if history_day_features and len(history_day_features) > 1:
        lag_feature_keys = [
            ("goes_frp_sum_at_candidate", "frp_sum"),
            ("goes_frp_max_at_candidate", "frp_max"),
            ("goes_active_frequency_at_candidate", "active_frequency"),
            ("goes_weighted_active_at_candidate", "weighted_active"),
            ("goes_frp_sum_5x5_max", "frp_sum_5x5_max"),
            ("goes_weighted_active_5x5_max", "weighted_active_5x5_max"),
        ]
        for base_name, source_key in lag_feature_keys:
            lag_values = []
            for lag, lag_features in enumerate(history_day_features):
                arr = np.asarray(lag_features[source_key])
                vals = arr[rr, cc].astype(np.float32)
                out[f"{base_name}_lag{lag}"] = vals
                lag_values.append(vals)
            stacked = np.vstack(lag_values[:available_goes_history_days])
            out[f"{base_name}_hist_mean"] = np.nanmean(stacked, axis=0).astype(np.float32)
            out[f"{base_name}_hist_max"] = np.nanmax(stacked, axis=0).astype(np.float32)
            out[f"{base_name}_hist_sum"] = np.nansum(stacked, axis=0).astype(np.float32)
        valid_history = history_day_features[:available_goes_history_days]
        out["goes_frp_total_log1p_hist_sum"] = float(np.nansum([f["goes_frp_total_log1p"] for f in valid_history]))
        out["goes_weighted_active_total_hist_sum"] = float(np.nansum([f["goes_weighted_active_total"] for f in valid_history]))

    weighted_row_delta = float(day_features["goes_weighted_active_centroid_row"]) - out["nearest_fire_row"].to_numpy(dtype=np.float32)
    weighted_col_delta = float(day_features["goes_weighted_active_centroid_col"]) - out["nearest_fire_col"].to_numpy(dtype=np.float32)
    out["goes_candidate_weighted_active_alignment"] = cosine_alignment(
        cand_row_delta,
        cand_col_delta,
        weighted_row_delta.astype(np.float32),
        weighted_col_delta.astype(np.float32),
    )
    return out


def enrich_candidates(
    input_path: Path,
    output_path: Path,
    goes_root: Path,
    chunksize: int,
    high_frp_percentile: float,
    history_days: int,
    allow_partial_history: bool,
) -> None:
    if output_path.exists():
        output_path.unlink()

    cache = GoesFrpCache(goes_root=goes_root)
    wrote_header = False
    total_rows = 0
    total_positive = 0

    for chunk_idx, chunk in enumerate(read_in_chunks(input_path, chunksize=chunksize), start=1):
        enriched_groups = []
        for (fire_id, date), group in chunk.groupby(["fire_id", "date"], sort=False):
            fire_id = str(fire_id)
            date = str(date)
            if allow_partial_history and "available_history_days" in group.columns:
                available_days = int(group["available_history_days"].iloc[0])
                available_days = max(1, min(history_days, available_days))
            else:
                available_days = history_days
            history_features = []
            for lag in range(history_days):
                if lag < available_days:
                    history_features.append(cache.get_day_features(fire_id, date_lag(date, lag), high_frp_percentile))
                else:
                    history_features.append(cache.empty_day_features())
            enriched_groups.append(
                add_goes_features(
                    group,
                    history_features[0],
                    history_features,
                    available_goes_history_days=available_days,
                )
            )

        enriched = pd.concat(enriched_groups, ignore_index=True)
        enriched.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        total_rows += len(enriched)
        total_positive += int(enriched["label_ignited_next_day"].sum())
        print(
            f"chunk={chunk_idx} rows={len(enriched)} total_rows={total_rows} "
            f"positive_rate={total_positive / total_rows:.6f}"
        )

    print(f"Wrote {output_path}")
    print(f"total_rows={total_rows} total_positive={total_positive} positive_rate={total_positive / total_rows:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Join day-d GOES FRP features onto event candidate CSV rows.")
    parser.add_argument("-mode", choices=["train", "val", "test"], required=True)
    parser.add_argument("--candidate-root", type=Path, default=Path(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--goes-root", type=Path, default=Path(DEFAULT_GOES_ROOT))
    parser.add_argument("--connectivity", type=int, default=8)
    parser.add_argument("--candidate-radius", type=float, default=5.0)
    parser.add_argument("--min-component-pixels", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--high-frp-percentile", type=float, default=90.0)
    parser.add_argument("--history-days", type=int, default=1, help="Number of GOES days ending at candidate date d to join. Use 2/4/6 to match VIIRS history candidates.")
    parser.add_argument(
        "--allow-partial-history",
        action="store_true",
        help="Read/write partial-history candidate files and zero-fill unavailable early GOES lags.",
    )
    args = parser.parse_args()

    if args.history_days < 1:
        raise ValueError("--history-days must be >= 1")
    suffix = f"{args.mode}_conn{args.connectivity}_r{radius_tag(args.candidate_radius)}_mincomp{args.min_component_pixels}{history_suffix(args.history_days, args.allow_partial_history)}"
    input_path = args.candidate_root / f"pred_event_candidates_{suffix}.csv"
    output_path = args.candidate_root / f"pred_event_candidates_{suffix}_goes_frp.csv"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    enrich_candidates(
        input_path=input_path,
        output_path=output_path,
        goes_root=args.goes_root,
        chunksize=args.chunksize,
        high_frp_percentile=args.high_frp_percentile,
        history_days=args.history_days,
        allow_partial_history=args.allow_partial_history,
    )


if __name__ == "__main__":
    main()
