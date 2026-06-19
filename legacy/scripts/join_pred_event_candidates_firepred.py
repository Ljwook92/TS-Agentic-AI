from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import ndimage

from dataset_gen_pred_goes_spatial import (
    CROP_OFFSET,
    CROP_SIZE,
    firepred_path_from_viirs,
    parse_timestamp_from_name,
    viirs_day_files,
)
from join_pred_event_candidates_goes_frp import (
    DEFAULT_CANDIDATE_ROOT,
    history_suffix,
    radius_tag,
)


FIREPRED_BANDS = {
    "ndvi": 0,
    "evi2": 1,
    "precip": 2,
    "wind_speed": 3,
    "wind_direction": 4,
    "tmin": 5,
    "tmax": 6,
    "erc": 7,
    "specific_humidity": 8,
    "slope": 9,
    "aspect": 10,
    "elevation": 11,
    "pdsi": 12,
    "landcover": 13,
    "forecast_precip": 14,
    "forecast_wind_speed": 15,
    "forecast_wind_direction": 16,
    "forecast_temperature": 17,
    "forecast_specific_humidity": 18,
}
LOCAL_MEAN_FEATURES = [
    "ndvi",
    "evi2",
    "precip",
    "wind_speed",
    "erc",
    "specific_humidity",
    "slope",
    "elevation",
    "pdsi",
    "forecast_precip",
    "forecast_wind_speed",
    "forecast_temperature",
    "forecast_specific_humidity",
]


def candidate_suffix(
    mode: str,
    connectivity: int,
    candidate_radius: float,
    min_component_pixels: int,
    history_days: int,
    allow_partial_history: bool,
) -> str:
    return (
        f"{mode}_conn{connectivity}_r{radius_tag(candidate_radius)}_"
        f"mincomp{min_component_pixels}{history_suffix(history_days, allow_partial_history)}"
    )


def local_mean(arr: np.ndarray, radius: int = 2) -> np.ndarray:
    size = 2 * radius + 1
    return ndimage.uniform_filter(arr.astype(np.float32), size=size, mode="nearest").astype(np.float32)


def direction_to_unit_vectors(direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Meteorological direction is where wind comes from; fire transport follows where it goes.
    theta_to = np.deg2rad(direction_deg.astype(np.float32) + 180.0)
    row = -np.cos(theta_to)
    col = np.sin(theta_to)
    return row.astype(np.float32), col.astype(np.float32)


def candidate_unit_vectors(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = group["candidate_row"].to_numpy(dtype=np.float32) - group["nearest_fire_row"].to_numpy(dtype=np.float32)
    col = group["candidate_col"].to_numpy(dtype=np.float32) - group["nearest_fire_col"].to_numpy(dtype=np.float32)
    distance = np.hypot(row, col).astype(np.float32)
    valid = distance > 0
    row_unit = np.zeros_like(row)
    col_unit = np.zeros_like(col)
    row_unit[valid] = row[valid] / distance[valid]
    col_unit[valid] = col[valid] / distance[valid]
    return row_unit, col_unit, distance


class FirePredCache:
    def __init__(self):
        self.file_cache: dict[str, dict[str, Path]] = {}
        self.day_cache: dict[tuple[str, str], tuple[np.ndarray, bool]] = {}
        self.active_fire_id: str | None = None

    def files_by_date(self, fire_id: str) -> dict[str, Path]:
        if fire_id in self.file_cache:
            return self.file_cache[fire_id]
        files: dict[str, Path] = {}
        for viirs_path in viirs_day_files(fire_id):
            timestamp = parse_timestamp_from_name(viirs_path.name)
            if timestamp is None:
                continue
            firepred_path = firepred_path_from_viirs(viirs_path)
            if firepred_path.exists():
                files[timestamp.date().isoformat()] = firepred_path
        self.file_cache[fire_id] = files
        return files

    def get(self, fire_id: str, date: str) -> tuple[np.ndarray, bool]:
        if fire_id != self.active_fire_id:
            self.day_cache.clear()
            self.active_fire_id = fire_id
        key = (fire_id, date)
        if key in self.day_cache:
            return self.day_cache[key]
        path = self.files_by_date(fire_id).get(date)
        if path is None:
            result = (np.zeros((len(FIREPRED_BANDS), CROP_SIZE, CROP_SIZE), dtype=np.float32), False)
            self.day_cache[key] = result
            return result
        window = Window(CROP_OFFSET, CROP_OFFSET, CROP_SIZE, CROP_SIZE)
        with rasterio.open(path) as src:
            arr = src.read(window=window).astype(np.float32, copy=False)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] < len(FIREPRED_BANDS):
            padded = np.zeros((len(FIREPRED_BANDS), CROP_SIZE, CROP_SIZE), dtype=np.float32)
            padded[: arr.shape[0]] = arr
            arr = padded
        else:
            arr = arr[: len(FIREPRED_BANDS)]
        result = (arr, True)
        self.day_cache[key] = result
        return result


def add_firepred_features(group: pd.DataFrame, arr: np.ndarray, available: bool) -> pd.DataFrame:
    out = group.copy()
    rr = out["candidate_row"].to_numpy(dtype=np.int64)
    cc = out["candidate_col"].to_numpy(dtype=np.int64)
    nr = out["nearest_fire_row"].to_numpy(dtype=np.int64)
    nc = out["nearest_fire_col"].to_numpy(dtype=np.int64)
    row_unit, col_unit, distance = candidate_unit_vectors(out)
    out["firepred_available"] = int(available)

    for name, band_idx in FIREPRED_BANDS.items():
        band = arr[band_idx]
        candidate_values = band[rr, cc].astype(np.float32)
        out[f"firepred_{name}_candidate"] = candidate_values
        if name in LOCAL_MEAN_FEATURES:
            out[f"firepred_{name}_5x5_mean"] = local_mean(band)[rr, cc]

    elevation = arr[FIREPRED_BANDS["elevation"]]
    elevation_delta = elevation[rr, cc] - elevation[nr, nc]
    out["firepred_elevation_delta_from_fire"] = elevation_delta.astype(np.float32)
    out["firepred_directional_elevation_gradient"] = (
        elevation_delta / np.maximum(distance, 1.0)
    ).astype(np.float32)

    landcover = np.rint(arr[FIREPRED_BANDS["landcover"]]).astype(np.int16)
    candidate_landcover = landcover[rr, cc]
    nearest_landcover = landcover[nr, nc]
    out["firepred_landcover_candidate"] = candidate_landcover
    out["firepred_landcover_nearest_fire"] = nearest_landcover
    out["firepred_landcover_same_as_fire"] = (candidate_landcover == nearest_landcover).astype(np.int8)
    out["firepred_landcover_transition"] = (
        candidate_landcover.astype(np.int32) * 100 + nearest_landcover.astype(np.int32)
    )

    for prefix in ["wind", "forecast_wind"]:
        speed = arr[FIREPRED_BANDS[f"{prefix}_speed"]][rr, cc].astype(np.float32)
        direction = arr[FIREPRED_BANDS[f"{prefix}_direction"]][rr, cc].astype(np.float32)
        wind_row, wind_col = direction_to_unit_vectors(direction)
        alignment = (row_unit * wind_row + col_unit * wind_col).astype(np.float32)
        cross_alignment = (row_unit * wind_col - col_unit * wind_row).astype(np.float32)
        out[f"firepred_{prefix}_candidate_alignment"] = alignment
        out[f"firepred_{prefix}_candidate_cross_alignment"] = cross_alignment
        out[f"firepred_{prefix}_candidate_push"] = (speed * alignment).astype(np.float32)
        out[f"firepred_{prefix}_candidate_crosswind"] = (speed * np.abs(cross_alignment)).astype(np.float32)

    aspect = arr[FIREPRED_BANDS["aspect"]][rr, cc].astype(np.float32)
    uphill_row, uphill_col = direction_to_unit_vectors(aspect)
    # Aspect points downslope; treating it as direction-from returns the opposite uphill vector.
    out["firepred_uphill_candidate_alignment"] = (
        row_unit * uphill_row + col_unit * uphill_col
    ).astype(np.float32)
    return out


def enrich(
    input_path: Path,
    output_path: Path,
    chunksize: int,
    limit_chunks: int | None,
) -> None:
    if output_path.exists():
        output_path.unlink()
    cache = FirePredCache()
    wrote_header = False
    total_rows = 0
    total_available = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize), start=1):
        if limit_chunks is not None and chunk_idx > limit_chunks:
            break
        enriched_groups = []
        for (fire_id, date), group in chunk.groupby(["fire_id", "date"], sort=False):
            arr, available = cache.get(str(fire_id), str(date))
            enriched_groups.append(add_firepred_features(group, arr, available))
            total_available += int(available) * len(group)
        enriched = pd.concat(enriched_groups, ignore_index=True)
        enriched.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        total_rows += len(enriched)
        print(
            f"chunk={chunk_idx} rows={len(enriched)} total_rows={total_rows} "
            f"firepred_available_rate={total_available / total_rows:.6f}"
        )
    print(f"Wrote {output_path}")
    print(
        f"total_rows={total_rows} firepred_available_rows={total_available} "
        f"availability_rate={total_available / total_rows if total_rows else np.nan:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join aligned FirePred weather, fuel, terrain, and land-cover candidate features."
    )
    parser.add_argument("-mode", choices=["train", "val", "test"], required=True)
    parser.add_argument("--candidate-root", type=Path, default=Path(DEFAULT_CANDIDATE_ROOT))
    parser.add_argument("--connectivity", type=int, default=8)
    parser.add_argument("--candidate-radius", type=float, default=5.0)
    parser.add_argument("--min-component-pixels", type=int, default=1)
    parser.add_argument("--history-days", type=int, default=1)
    parser.add_argument("--allow-partial-history", action="store_true")
    parser.add_argument("--input-variant", default="goes_frp_motion")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--limit-chunks", type=int, default=None)
    args = parser.parse_args()

    suffix = candidate_suffix(
        args.mode,
        args.connectivity,
        args.candidate_radius,
        args.min_component_pixels,
        args.history_days,
        args.allow_partial_history,
    )
    input_path = args.candidate_root / f"pred_event_candidates_{suffix}_{args.input_variant}.csv"
    output_path = args.candidate_root / f"pred_event_candidates_{suffix}_{args.input_variant}_firepred.csv"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.limit_chunks is not None and args.limit_chunks < 1:
        raise ValueError("--limit-chunks must be >= 1")
    enrich(input_path, output_path, args.chunksize, args.limit_chunks)


if __name__ == "__main__":
    main()
