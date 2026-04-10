from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from support.path_config import get_code_root, get_raw_data_root, get_task_dataset_root


DEFAULT_GOES_ROOT = "/home/jlc3q/data/GOES_clipped_tif_common_wgs84"
RAW_DATA_DIR = Path(get_raw_data_root())
DATASET_DIR = Path(get_task_dataset_root("pred"))
ROI_DIR = get_code_root() / "legacy" / "roi"

MASK_DIR_NAMES = ("mask_fixed", "mask")
FRP_DIR_NAMES = ("frp_fixed", "frp")
TIFF_SUFFIXES = {".tif", ".tiff"}

FEATURE_NAMES = [
    "active_pixel_count",
    "mask_coverage",
    "frp_mean",
    "frp_max",
]
FIRE_MASK_CODES = set(range(10, 16)) | set(range(20, 26)) | set(range(30, 36))


def default_chunk_size(ts_length: int) -> int:
    env_value = os.environ.get("TS_SATFIRE_PRED_GOES_SUBDAILY_CHUNK_SIZE")
    if env_value:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    if ts_length >= 8:
        return 2
    if ts_length >= 6:
        return 3
    return 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-bin sub-daily GOES features aligned to pred windows."
    )
    parser.add_argument("-mode", type=str, choices=["train", "val", "test", "merge_train", "merge_val"])
    parser.add_argument("-ts", type=int, help="Length of TS")
    parser.add_argument("-it", type=int, help="Interval")
    parser.add_argument("--goes-root", default=DEFAULT_GOES_ROOT, help="Root directory of clipped GOES tif files.")
    parser.add_argument("--bin-minutes", type=int, default=15, help="Fixed sub-daily bin size in minutes.")
    parser.add_argument("-limit", type=int, default=None, help="Optional limit on number of fires to process")
    parser.add_argument("-start", type=int, default=0, help="Optional start index into the filtered fire list")
    return parser.parse_args()


def load_roi_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_years = []
    for year in ["2017", "2018", "2019", "2020"]:
        train_years.append(pd.read_csv(ROI_DIR / f"us_fire_{year}_out_new.csv"))
    test_years = [pd.read_csv(ROI_DIR / "us_fire_2021_out_new.csv")]
    return pd.concat(train_years, ignore_index=True), pd.concat(test_years, ignore_index=True)


def resolve_locations(mode: str) -> list[str]:
    train_df, test_df = load_roi_tables()
    val_ids = [
        "20568194", "20701026", "20562846", "20700973", "24462610", "24462788", "24462753",
        "24103571", "21998313", "21751303", "22141596", "21999381", "23301962", "22712904", "22713339",
    ]

    train_df = train_df.sort_values(by=["Id"])
    train_df["Id"] = train_df["Id"].astype(str)
    train_split = train_df[~train_df.Id.isin(val_ids)]
    val_split = train_df[train_df.Id.isin(val_ids)]

    test_df = test_df.sort_values(by=["Id"])
    test_df["Id"] = test_df["Id"].astype(str)

    if mode == "train":
        return train_split["Id"].tolist()
    if mode == "val":
        return val_split["Id"].tolist()
    return test_df["Id"].tolist()


def has_prediction_inputs(location_id: str) -> bool:
    location_root = RAW_DATA_DIR / location_id
    viirs_day_dir = location_root / "VIIRS_Day"
    firepred_dir = location_root / "FirePred"
    return (
        viirs_day_dir.is_dir()
        and firepred_dir.is_dir()
        and any(path.suffix.lower() in TIFF_SUFFIXES for path in viirs_day_dir.iterdir())
        and any(path.suffix.lower() in TIFF_SUFFIXES for path in firepred_dir.iterdir())
    )


def parse_timestamp_from_name(name: str) -> datetime | None:
    stem = Path(name).stem
    patterns = [
        (r"(?<!\d)(20\d{6}T\d{6})(?!\d)", "%Y%m%dT%H%M%S"),
        (r"(?<!\d)(20\d{12})(?!\d)", "%Y%m%d%H%M%S"),
        (r"(?<!\d)(20\d{8})(?!\d)", "%Y%m%d%H"),
        (r"(?<!\d)(20\d{6})(?!\d)", "%Y%m%d"),
        (r"(?<!\d)s(20\d{2})(\d{3})(\d{6})(?!\d)", None),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, stem)
        if not match:
            continue
        if fmt is not None:
            try:
                return datetime.strptime(match.group(1), fmt)
            except ValueError:
                continue
        year, doy, hms = match.groups()
        try:
            return datetime.strptime(f"{year}{doy}{hms}", "%Y%j%H%M%S")
        except ValueError:
            continue
    try:
        return datetime.strptime(stem.replace("_VIIRS_Day", ""), "%Y-%m-%d")
    except ValueError:
        return None


def find_event_dir(goes_root: Path, event_id: str) -> Path | None:
    direct = goes_root / event_id
    if direct.is_dir():
        return direct
    for year_dir in sorted(path for path in goes_root.iterdir() if path.is_dir()):
        candidate = year_dir / event_id
        if candidate.is_dir():
            return candidate
    matches = [path for path in goes_root.rglob(event_id) if path.is_dir()]
    return matches[0] if matches else None


def viirs_day_files(location_id: str) -> list[Path]:
    viirs_dir = RAW_DATA_DIR / location_id / "VIIRS_Day"
    return sorted(path for path in viirs_dir.iterdir() if path.suffix.lower() in TIFF_SUFFIXES)


def output_paths(mode: str, ts_length: int, interval: int, location_id: str | None = None) -> Path:
    target_dir = DATASET_DIR / f"dataset_{mode}"
    target_dir.mkdir(parents=True, exist_ok=True)
    if mode == "test":
        assert location_id is not None
        return target_dir / f"pred_{location_id}_goes_subdaily_seqtoseql_{ts_length}i_{interval}.npy"
    return target_dir / f"pred_{mode}_goes_subdaily_seqtoseq_alll_{ts_length}i_{interval}.npy"


def chunk_output_path(mode: str, ts_length: int, interval: int, start: int, end: int) -> Path:
    target_dir = DATASET_DIR / f"dataset_{mode}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"pred_{mode}_goes_subdaily_seqtoseq_alll_{ts_length}i_{interval}_part_{start}_{end}.npy"


def metadata_path(ts_length: int, interval: int, bins_per_day: int) -> Path:
    return DATASET_DIR / f"pred_goes_subdaily_feature_names_{ts_length}i_{interval}_{bins_per_day}b.csv"


def go_path_iter(event_dir: Path, dir_names: tuple[str, ...]) -> list[Path]:
    for dirname in dir_names:
        subdir = event_dir / dirname
        if subdir.is_dir():
            return sorted(path for path in subdir.iterdir() if path.suffix.lower() in TIFF_SUFFIXES)
    return []


def build_day_bin_store(
    event_dir: Path,
    bin_minutes: int,
) -> tuple[dict[str, dict[int, list[Path]]], dict[str, dict[int, list[Path]]]]:
    mask_store: dict[str, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))
    frp_store: dict[str, dict[int, list[Path]]] = defaultdict(lambda: defaultdict(list))

    for path in go_path_iter(event_dir, MASK_DIR_NAMES):
        timestamp = parse_timestamp_from_name(path.name)
        if timestamp is None:
            continue
        bin_idx = (timestamp.hour * 60 + timestamp.minute) // bin_minutes
        mask_store[timestamp.date().isoformat()][bin_idx].append(path)

    for path in go_path_iter(event_dir, FRP_DIR_NAMES):
        timestamp = parse_timestamp_from_name(path.name)
        if timestamp is None:
            continue
        bin_idx = (timestamp.hour * 60 + timestamp.minute) // bin_minutes
        frp_store[timestamp.date().isoformat()][bin_idx].append(path)

    return mask_store, frp_store


def mask_bin_features(paths: list[Path]) -> tuple[float, float]:
    if not paths:
        return 0.0, 0.0
    active_counts = []
    coverages = []
    for path in paths:
        with rasterio.open(path) as src:
            arr = np.nan_to_num(src.read(1), nan=0.0)
        fire_mask = np.isin(arr, list(FIRE_MASK_CODES))
        active = float(fire_mask.sum())
        coverage = float(fire_mask.mean())
        active_counts.append(active)
        coverages.append(coverage)
    return float(max(active_counts)), float(max(coverages))


def frp_bin_features(paths: list[Path]) -> tuple[float, float]:
    if not paths:
        return 0.0, 0.0
    means = []
    maxima = []
    for path in paths:
        with rasterio.open(path) as src:
            arr = np.nan_to_num(src.read(1), nan=0.0)
        positive = arr[arr > 0]
        if positive.size == 0:
            means.append(0.0)
            maxima.append(0.0)
        else:
            means.append(float(positive.mean()))
            maxima.append(float(positive.max()))
    return float(np.mean(means)), float(max(maxima))


def day_subdaily_matrix(
    day_key: str,
    mask_store: dict[str, dict[int, list[Path]]],
    frp_store: dict[str, dict[int, list[Path]]],
    bins_per_day: int,
) -> np.ndarray:
    day_matrix = np.zeros((bins_per_day, len(FEATURE_NAMES)), dtype=np.float32)
    day_mask_bins = mask_store.get(day_key, {})
    day_frp_bins = frp_store.get(day_key, {})
    for bin_idx in range(bins_per_day):
        active_pixel_count, mask_coverage = mask_bin_features(day_mask_bins.get(bin_idx, []))
        frp_mean, frp_max = frp_bin_features(day_frp_bins.get(bin_idx, []))
        day_matrix[bin_idx] = np.array(
            [active_pixel_count, mask_coverage, frp_mean, frp_max],
            dtype=np.float32,
        )
    return day_matrix


def generate_event_samples(
    location_id: str,
    goes_root: Path,
    ts_length: int,
    interval: int,
    bin_minutes: int,
) -> np.ndarray:
    files = viirs_day_files(location_id)
    bins_per_day = math.ceil((24 * 60) / bin_minutes)
    if not files:
        return np.zeros((0, ts_length, bins_per_day, len(FEATURE_NAMES)), dtype=np.float32)

    event_dir = find_event_dir(goes_root, location_id)
    if event_dir is None:
        return np.zeros((max(len(files) - ts_length, 0), ts_length, bins_per_day, len(FEATURE_NAMES)), dtype=np.float32)
    mask_store, frp_store = build_day_bin_store(event_dir, bin_minutes=bin_minutes)

    sample_rows: list[np.ndarray] = []
    for i in range(0, len(files), interval):
        if i + ts_length >= len(files):
            break
        per_step: list[np.ndarray] = []
        for j in range(ts_length):
            timestamp = parse_timestamp_from_name(files[i + j].name)
            if timestamp is None:
                per_step.append(np.zeros((bins_per_day, len(FEATURE_NAMES)), dtype=np.float32))
                continue
            per_step.append(
                day_subdaily_matrix(
                    timestamp.date().isoformat(),
                    mask_store=mask_store,
                    frp_store=frp_store,
                    bins_per_day=bins_per_day,
                )
            )
        sample_rows.append(np.stack(per_step, axis=0))

    if not sample_rows:
        return np.zeros((0, ts_length, bins_per_day, len(FEATURE_NAMES)), dtype=np.float32)
    return np.stack(sample_rows, axis=0).astype(np.float32)


def write_feature_names(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "feature_name"])
        for idx, name in enumerate(FEATURE_NAMES):
            writer.writerow([idx, name])


def merge_chunk_files(mode: str, ts_length: int, interval: int) -> None:
    if mode not in {"train", "val"}:
        raise ValueError("Chunk merge is only supported for train/val.")

    target_dir = DATASET_DIR / f"dataset_{mode}"
    chunk_paths = sorted(target_dir.glob(f"pred_{mode}_goes_subdaily_seqtoseq_alll_{ts_length}i_{interval}_part_*_*.npy"))
    if not chunk_paths:
        raise FileNotFoundError(f"No GOES sub-daily chunk files found for mode={mode}, ts={ts_length}, interval={interval}.")

    arrays = [np.load(path, mmap_mode="r") for path in chunk_paths]
    total_rows = sum(arr.shape[0] for arr in arrays)
    out_shape = (total_rows, *arrays[0].shape[1:])

    out_path = output_paths(mode=mode, ts_length=ts_length, interval=interval)
    if out_path.exists():
        out_path.unlink()

    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=out_shape)
    offset = 0
    for arr in arrays:
        next_offset = offset + arr.shape[0]
        out[offset:next_offset] = arr
        offset = next_offset
    out.flush()
    print(f"Merged {len(chunk_paths)} GOES sub-daily chunk files into {out_path}")


def generate_train_val_chunks(
    mode: str,
    locations: list[str],
    goes_root: Path,
    ts_length: int,
    interval: int,
    bin_minutes: int,
) -> None:
    chunk_size = default_chunk_size(ts_length)
    total = len(locations)
    print(
        f"Auto chunking GOES sub-daily {mode} split: {total} fires, "
        f"chunk_size={chunk_size}, ts_length={ts_length}, interval={interval}, bin_minutes={bin_minutes}"
    )

    for start in range(0, total, chunk_size):
        chunk_locations = locations[start:start + chunk_size]
        end = start + len(chunk_locations)
        out_path = chunk_output_path(mode=mode, ts_length=ts_length, interval=interval, start=start, end=end)
        chunk_rows: list[np.ndarray] = []
        for location in tqdm(chunk_locations, desc=f"GOES sub-daily {mode} chunk [{start}:{end}]", unit="fire"):
            feats = generate_event_samples(
                location_id=location,
                goes_root=goes_root,
                ts_length=ts_length,
                interval=interval,
                bin_minutes=bin_minutes,
            )
            if feats.shape[0] > 0:
                chunk_rows.append(feats)
        if not chunk_rows:
            print(f"Skipping empty GOES sub-daily chunk [{start}:{end}]")
            continue
        merged = np.concatenate(chunk_rows, axis=0).astype(np.float32)
        np.save(out_path, merged)
        print(f"Wrote GOES sub-daily chunk [{start}:{end}] with shape {merged.shape} to {out_path}")

    merge_chunk_files(mode=mode, ts_length=ts_length, interval=interval)


def main() -> None:
    args = parse_args()
    mode = args.mode
    ts_length = args.ts
    interval = args.it
    bin_minutes = args.bin_minutes
    bins_per_day = math.ceil((24 * 60) / bin_minutes)
    goes_root = Path(args.goes_root)

    if mode == "merge_train":
        merge_chunk_files("train", ts_length=ts_length, interval=interval)
        return
    if mode == "merge_val":
        merge_chunk_files("val", ts_length=ts_length, interval=interval)
        return

    locations = resolve_locations(mode)
    filtered = [location for location in locations if has_prediction_inputs(location)]
    filtered = filtered[max(args.start, 0):]
    if args.limit is not None:
        filtered = filtered[:args.limit]
    if not filtered:
        raise RuntimeError(f"No valid prediction inputs found for mode={mode} after filtering.")

    write_feature_names(metadata_path(ts_length=ts_length, interval=interval, bins_per_day=bins_per_day))

    if mode == "test":
        for location in tqdm(filtered, desc="Generating GOES sub-daily test features", unit="fire"):
            feats = generate_event_samples(
                location_id=location,
                goes_root=goes_root,
                ts_length=ts_length,
                interval=interval,
                bin_minutes=bin_minutes,
            )
            out_path = output_paths(mode=mode, ts_length=ts_length, interval=interval, location_id=location)
            np.save(out_path, feats.astype(np.float32))
            print(f"{location}: wrote {feats.shape} to {out_path}")
        return

    if args.start > 0 or args.limit is not None:
        end = args.start + len(filtered)
        chunk_rows: list[np.ndarray] = []
        for location in tqdm(filtered, desc=f"Generating GOES sub-daily {mode} features", unit="fire"):
            feats = generate_event_samples(
                location_id=location,
                goes_root=goes_root,
                ts_length=ts_length,
                interval=interval,
                bin_minutes=bin_minutes,
            )
            if feats.shape[0] > 0:
                chunk_rows.append(feats)
        if not chunk_rows:
            raise RuntimeError(f"No GOES sub-daily features were generated for mode={mode}.")
        merged = np.concatenate(chunk_rows, axis=0).astype(np.float32)
        out_path = chunk_output_path(mode=mode, ts_length=ts_length, interval=interval, start=args.start, end=end)
        np.save(out_path, merged)
        print(f"Wrote {merged.shape} to {out_path}")
        return

    generate_train_val_chunks(
        mode=mode,
        locations=filtered,
        goes_root=goes_root,
        ts_length=ts_length,
        interval=interval,
        bin_minutes=bin_minutes,
    )


if __name__ == "__main__":
    main()
