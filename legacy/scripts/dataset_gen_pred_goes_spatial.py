from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from tqdm import tqdm

from support.path_config import get_code_root, get_raw_data_root, get_task_dataset_root


DEFAULT_GOES_ROOT = "/home/jlc3q/data/GOES_clipped_tif_common_wgs84"
RAW_DATA_DIR = Path(get_raw_data_root())
DATASET_DIR = Path(get_task_dataset_root("pred"))
ROI_DIR = get_code_root() / "legacy" / "roi"

MASK_DIR_NAMES = ("mask_fixed", "mask")
FRP_DIR_NAMES = ("frp_fixed", "frp")
TIFF_SUFFIXES = {".tif", ".tiff"}
FIRE_MASK_CODES = set(range(10, 16)) | set(range(20, 26)) | set(range(30, 36))

CROP_OFFSET = 128
CROP_SIZE = 256

FEATURE_NAMES = [
    "daily_active",
    "active_frequency",
    "cumulative_active",
    "new_active",
    "frp_sum_log1p",
    "frp_max_log1p",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate spatial GOES-R FDCF maps aligned to TS-SatFire pred VIIRS windows."
    )
    parser.add_argument("-mode", type=str, choices=["train", "val", "test", "merge_train", "merge_val"], required=True)
    parser.add_argument("-ts", type=int, required=True, help="Length of VIIRS input time series")
    parser.add_argument("-it", type=int, required=True, help="VIIRS sampling interval")
    parser.add_argument("--goes-root", default=DEFAULT_GOES_ROOT, help="Root directory of clipped GOES tif files")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="Output dtype")
    parser.add_argument("-limit", type=int, default=None, help="Optional limit on number of fires to process")
    parser.add_argument("-start", type=int, default=0, help="Optional start index into the filtered fire list")
    return parser.parse_args()


def default_chunk_size(ts_length: int) -> int:
    env_value = os.environ.get("TS_SATFIRE_PRED_GOES_SPATIAL_CHUNK_SIZE")
    if env_value:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    if ts_length >= 6:
        return 1
    return 2


def load_roi_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_years = [pd.read_csv(ROI_DIR / f"us_fire_{year}_out_new.csv") for year in ["2017", "2018", "2019", "2020"]]
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


def paths_in_first_existing_dir(event_dir: Path, names: tuple[str, ...]) -> list[Path]:
    for name in names:
        subdir = event_dir / name
        if subdir.is_dir():
            return sorted(path for path in subdir.iterdir() if path.suffix.lower() in TIFF_SUFFIXES)
    return []


def collect_goes_files_by_day(event_dir: Path) -> dict[str, dict[str, list[Path]]]:
    by_day: dict[str, dict[str, list[Path]]] = defaultdict(lambda: {"mask": [], "frp": []})
    for kind, names in [("mask", MASK_DIR_NAMES), ("frp", FRP_DIR_NAMES)]:
        for path in paths_in_first_existing_dir(event_dir, names):
            timestamp = parse_timestamp_from_name(path.name)
            if timestamp is not None:
                by_day[timestamp.date().isoformat()][kind].append(path)
    return by_day


def viirs_day_files(location_id: str) -> list[Path]:
    viirs_dir = RAW_DATA_DIR / location_id / "VIIRS_Day"
    return sorted(path for path in viirs_dir.iterdir() if path.suffix.lower() in TIFF_SUFFIXES)


def firepred_path_from_viirs(viirs_path: Path) -> Path:
    return Path(str(viirs_path).replace("/VIIRS_Day/", "/FirePred/").replace("_VIIRS_Day.tif", "_FirePred.tif"))


def crop_profile(reference_path: Path) -> dict:
    window = Window(CROP_OFFSET, CROP_OFFSET, CROP_SIZE, CROP_SIZE)
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            height=CROP_SIZE,
            width=CROP_SIZE,
            transform=src.window_transform(window),
            dtype="float32",
            nodata=0,
        )
    return profile


def reproject_to_crop(src_array: np.ndarray, src_profile: dict, dst_profile: dict, resampling: Resampling) -> np.ndarray:
    dst = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    reproject(
        source=src_array.astype(np.float32, copy=False),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        src_nodata=0,
        dst_nodata=0,
        resampling=resampling,
    )
    return np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0)


def daily_maps(day_key: str, goes_by_day: dict[str, dict[str, list[Path]]], dst_profile: dict) -> dict[str, np.ndarray]:
    bucket = goes_by_day.get(day_key, {"mask": [], "frp": []})
    mask_count = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    frp_sum = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
    frp_max = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)

    for path in bucket["mask"]:
        with rasterio.open(path) as src:
            active = np.isin(np.nan_to_num(src.read(1), nan=0.0), list(FIRE_MASK_CODES)).astype(np.float32)
            projected = reproject_to_crop(active, src.profile, dst_profile, Resampling.nearest)
        mask_count += (projected > 0).astype(np.float32)

    for path in bucket["frp"]:
        with rasterio.open(path) as src:
            frp = np.nan_to_num(src.read(1), nan=0.0, posinf=0.0, neginf=0.0)
            frp = np.where(frp > 0, frp, 0).astype(np.float32)
            projected = reproject_to_crop(frp, src.profile, dst_profile, Resampling.bilinear)
        frp_sum += projected
        frp_max = np.maximum(frp_max, projected)

    n_masks = max(len(bucket["mask"]), 1)
    daily_active = (mask_count > 0).astype(np.float32)
    active_frequency = mask_count / float(n_masks)
    return {
        "daily_active": daily_active,
        "active_frequency": active_frequency,
        "frp_sum_log1p": np.log1p(frp_sum),
        "frp_max_log1p": np.log1p(frp_max),
    }


def generate_event_samples(location_id: str, goes_root: Path, ts_length: int, interval: int) -> np.ndarray:
    files = viirs_day_files(location_id)
    sample_count = max(len(files) - ts_length, 0)
    if sample_count == 0:
        return np.zeros((0, len(FEATURE_NAMES), ts_length, CROP_SIZE, CROP_SIZE), dtype=np.float32)

    ref_firepred = firepred_path_from_viirs(files[0])
    if not ref_firepred.exists():
        return np.zeros((sample_count, len(FEATURE_NAMES), ts_length, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    dst_profile = crop_profile(ref_firepred)

    event_dir = find_event_dir(goes_root, location_id)
    goes_by_day = collect_goes_files_by_day(event_dir) if event_dir else {}
    day_cache: dict[str, dict[str, np.ndarray]] = {}

    sample_rows: list[np.ndarray] = []
    for i in range(0, len(files), interval):
        if i + ts_length >= len(files):
            break
        cumulative = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        per_step: list[np.ndarray] = []
        for j in range(ts_length):
            timestamp = parse_timestamp_from_name(files[i + j].name)
            day_key = timestamp.date().isoformat() if timestamp else ""
            if day_key not in day_cache:
                day_cache[day_key] = daily_maps(day_key, goes_by_day, dst_profile)

            day = day_cache[day_key]
            daily_active = day["daily_active"]
            new_active = np.logical_and(daily_active > 0, cumulative == 0).astype(np.float32)
            cumulative = np.logical_or(cumulative > 0, daily_active > 0).astype(np.float32)

            per_step.append(
                np.stack(
                    [
                        daily_active,
                        day["active_frequency"],
                        cumulative,
                        new_active,
                        day["frp_sum_log1p"],
                        day["frp_max_log1p"],
                    ],
                    axis=0,
                )
            )
        sample_rows.append(np.stack(per_step, axis=1))

    if not sample_rows:
        return np.zeros((0, len(FEATURE_NAMES), ts_length, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    return np.stack(sample_rows, axis=0).astype(np.float32)


def output_path(mode: str, ts_length: int, interval: int, location_id: str | None = None) -> Path:
    target_dir = DATASET_DIR / f"dataset_{mode}"
    target_dir.mkdir(parents=True, exist_ok=True)
    if mode == "test":
        assert location_id is not None
        return target_dir / f"pred_{location_id}_goes_spatial_seqtoseql_{ts_length}i_{interval}.npy"
    return target_dir / f"pred_{mode}_goes_spatial_seqtoseq_alll_{ts_length}i_{interval}.npy"


def chunk_output_path(mode: str, ts_length: int, interval: int, start: int, end: int) -> Path:
    target_dir = DATASET_DIR / f"dataset_{mode}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"pred_{mode}_goes_spatial_seqtoseq_alll_{ts_length}i_{interval}_part_{start}_{end}.npy"


def metadata_path(ts_length: int, interval: int) -> Path:
    return DATASET_DIR / f"pred_goes_spatial_feature_names_{ts_length}i_{interval}.csv"


def write_feature_names(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_index", "feature_name"])
        for idx, name in enumerate(FEATURE_NAMES):
            writer.writerow([idx, name])


def chunk_sort_key(path: Path) -> tuple[int, int]:
    match = re.search(r"_part_(\d+)_(\d+)\.npy$", path.name)
    if not match:
        raise ValueError(f"Cannot parse chunk indices from {path}")
    return int(match.group(1)), int(match.group(2))


def merge_chunk_files(mode: str, ts_length: int, interval: int, dtype: str) -> None:
    if mode not in {"train", "val"}:
        raise ValueError("Chunk merge is only supported for train/val.")

    target_dir = DATASET_DIR / f"dataset_{mode}"
    chunk_paths = sorted(
        target_dir.glob(f"pred_{mode}_goes_spatial_seqtoseq_alll_{ts_length}i_{interval}_part_*_*.npy"),
        key=chunk_sort_key,
    )
    if not chunk_paths:
        raise FileNotFoundError(f"No GOES spatial chunk files found for mode={mode}, ts={ts_length}, interval={interval}.")

    arrays = [np.load(path, mmap_mode="r") for path in chunk_paths]
    total_rows = sum(arr.shape[0] for arr in arrays)
    out_shape = (total_rows, *arrays[0].shape[1:])

    out = np.lib.format.open_memmap(output_path(mode, ts_length, interval), mode="w+", dtype=np.dtype(dtype), shape=out_shape)
    offset = 0
    for arr in arrays:
        next_offset = offset + arr.shape[0]
        out[offset:next_offset] = arr.astype(dtype, copy=False)
        offset = next_offset
    out.flush()
    print(f"Merged {len(chunk_paths)} GOES spatial chunk files into {output_path(mode, ts_length, interval)} with shape {out_shape}")


def generate_train_val_chunks(mode: str, locations: list[str], goes_root: Path, ts_length: int, interval: int, dtype: str) -> None:
    chunk_size = default_chunk_size(ts_length)
    total = len(locations)
    print(f"Auto chunking GOES spatial {mode}: {total} fires, chunk_size={chunk_size}, ts_length={ts_length}, interval={interval}")

    for start in range(0, total, chunk_size):
        chunk_locations = locations[start:start + chunk_size]
        end = start + len(chunk_locations)
        rows: list[np.ndarray] = []
        for location in tqdm(chunk_locations, desc=f"GOES spatial {mode} chunk [{start}:{end}]", unit="fire"):
            feats = generate_event_samples(location, goes_root, ts_length, interval)
            if feats.shape[0] > 0:
                rows.append(feats)
        if not rows:
            print(f"Skipping empty GOES spatial chunk [{start}:{end}]")
            continue
        merged = np.concatenate(rows, axis=0).astype(dtype, copy=False)
        out = chunk_output_path(mode, ts_length, interval, start, end)
        np.save(out, merged)
        print(f"Wrote GOES spatial chunk [{start}:{end}] with shape {merged.shape} to {out}")

    merge_chunk_files(mode, ts_length, interval, dtype)


def main() -> None:
    args = parse_args()
    goes_root = Path(args.goes_root)

    if args.mode == "merge_train":
        merge_chunk_files("train", args.ts, args.it, args.dtype)
        return
    if args.mode == "merge_val":
        merge_chunk_files("val", args.ts, args.it, args.dtype)
        return

    locations = resolve_locations(args.mode)
    filtered = [location for location in locations if has_prediction_inputs(location)]
    filtered = filtered[max(args.start, 0):]
    if args.limit is not None:
        filtered = filtered[:args.limit]
    if not filtered:
        raise RuntimeError(f"No valid prediction inputs found for mode={args.mode}.")

    write_feature_names(metadata_path(args.ts, args.it))

    if args.mode == "test":
        for location in tqdm(filtered, desc="Generating GOES spatial test features", unit="fire"):
            feats = generate_event_samples(location, goes_root, args.ts, args.it).astype(args.dtype, copy=False)
            out = output_path(args.mode, args.ts, args.it, location)
            np.save(out, feats)
            print(f"{location}: wrote {feats.shape} to {out}")
        return

    if args.start > 0 or args.limit is not None:
        rows = []
        for location in tqdm(filtered, desc=f"Generating GOES spatial {args.mode} features", unit="fire"):
            feats = generate_event_samples(location, goes_root, args.ts, args.it)
            if feats.shape[0] > 0:
                rows.append(feats)
        if not rows:
            raise RuntimeError("No valid GOES spatial sequences were generated.")
        end = args.start + len(filtered)
        merged = np.concatenate(rows, axis=0).astype(args.dtype, copy=False)
        out = chunk_output_path(args.mode, args.ts, args.it, args.start, end)
        np.save(out, merged)
        print(f"Wrote {merged.shape} to {out}")
        return

    generate_train_val_chunks(args.mode, filtered, goes_root, args.ts, args.it, args.dtype)


if __name__ == "__main__":
    main()
