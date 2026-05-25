from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset_gen_pred_goes_spatial import (
    CROP_OFFSET,
    CROP_SIZE,
    DATASET_DIR,
    TIFF_SUFFIXES,
    has_prediction_inputs,
    parse_timestamp_from_name,
    resolve_locations,
    viirs_day_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create front-buffer constrained GOES spatial tensors from existing GOES spatial pred data."
    )
    parser.add_argument("-mode", type=str, choices=["train", "val", "test", "merge_train", "merge_val"], required=True)
    parser.add_argument("-ts", type=int, required=True)
    parser.add_argument("-it", type=int, required=True)
    parser.add_argument("--buffer-pixels", type=int, default=12)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--exclude-current-fire", action="store_true")
    parser.add_argument("-limit", type=int, default=None)
    parser.add_argument("-start", type=int, default=0)
    return parser.parse_args()


def default_chunk_size(ts_length: int) -> int:
    if ts_length >= 6:
        return 1
    return 2


def source_path(mode: str, ts_length: int, interval: int, location_id: str | None = None) -> Path:
    target_dir = DATASET_DIR / f"dataset_{mode}"
    if mode == "test":
        assert location_id is not None
        return target_dir / f"pred_{location_id}_goes_spatial_seqtoseql_{ts_length}i_{interval}.npy"
    return target_dir / f"pred_{mode}_goes_spatial_seqtoseq_alll_{ts_length}i_{interval}.npy"


def output_path(mode: str, ts_length: int, interval: int, location_id: str | None = None) -> Path:
    target_dir = DATASET_DIR / f"dataset_{mode}"
    target_dir.mkdir(parents=True, exist_ok=True)
    if mode == "test":
        assert location_id is not None
        return target_dir / f"pred_{location_id}_goes_spatial_frontbuf_seqtoseql_{ts_length}i_{interval}.npy"
    return target_dir / f"pred_{mode}_goes_spatial_frontbuf_seqtoseq_alll_{ts_length}i_{interval}.npy"


def chunk_output_path(mode: str, ts_length: int, interval: int, start: int, end: int) -> Path:
    target_dir = DATASET_DIR / f"dataset_{mode}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"pred_{mode}_goes_spatial_frontbuf_seqtoseq_alll_{ts_length}i_{interval}_part_{start}_{end}.npy"


def chunk_sort_key(path: Path) -> tuple[int, int]:
    match = re.search(r"_part_(\d+)_(\d+)\.npy$", path.name)
    if not match:
        raise ValueError(f"Cannot parse chunk indices from {path}")
    return int(match.group(1)), int(match.group(2))


def read_fire_mask(viirs_path: Path, include_ba: bool = False) -> np.ndarray:
    with rasterio.open(viirs_path) as src:
        af = src.read(7, window=((CROP_OFFSET, CROP_OFFSET + CROP_SIZE), (CROP_OFFSET, CROP_OFFSET + CROP_SIZE)))
        fire = np.nan_to_num(af, nan=0.0, posinf=0.0, neginf=0.0) > 0
        if include_ba and src.count >= 8:
            ba = src.read(8, window=((CROP_OFFSET, CROP_OFFSET + CROP_SIZE), (CROP_OFFSET, CROP_OFFSET + CROP_SIZE)))
            fire = np.logical_or(fire, np.nan_to_num(ba, nan=0.0, posinf=0.0, neginf=0.0) > 0)
    return fire[:CROP_SIZE, :CROP_SIZE]


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    x = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]
    y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[0, 0].numpy() > 0


def event_front_masks(location_id: str, ts_length: int, interval: int, buffer_pixels: int, exclude_current_fire: bool) -> np.ndarray:
    files = viirs_day_files(location_id)
    masks: list[np.ndarray] = []
    af_acc = np.zeros((CROP_SIZE, CROP_SIZE), dtype=bool)
    new_base = af_acc.copy()

    for i in range(0, len(files), interval):
        if i + ts_length >= len(files):
            break
        per_step: list[np.ndarray] = []
        for j in range(ts_length + 1):
            current_fire = read_fire_mask(files[i + j], include_ba=False)
            af_acc = np.logical_or(af_acc, current_fire)
            if j == interval - 1:
                new_base = af_acc.copy()
            if j < ts_length:
                front_buffer = dilate_mask(af_acc, buffer_pixels)
                if exclude_current_fire:
                    front_buffer = np.logical_and(front_buffer, ~af_acc)
                per_step.append(front_buffer.astype(np.float32))
        af_acc = new_base.copy()
        masks.append(np.stack(per_step, axis=0))

    if not masks:
        return np.zeros((0, ts_length, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    return np.stack(masks, axis=0).astype(np.float32)


def event_sample_count(location_id: str, ts_length: int, interval: int) -> int:
    files = viirs_day_files(location_id)
    return sum(1 for i in range(0, len(files), interval) if i + ts_length < len(files))


def cumulative_offsets(locations: list[str], ts_length: int, interval: int) -> dict[str, int]:
    offsets = {}
    offset = 0
    for location in locations:
        offsets[location] = offset
        offset += event_sample_count(location, ts_length, interval)
    return offsets


def build_event_frontbuf(
    location_id: str,
    mode: str,
    ts_length: int,
    interval: int,
    buffer_pixels: int,
    exclude_current_fire: bool,
    source_offset: int | None = None,
) -> np.ndarray:
    goes_path = source_path(mode, ts_length, interval, location_id if mode == "test" else None)
    goes = np.load(goes_path, mmap_mode="r")
    masks = event_front_masks(location_id, ts_length, interval, buffer_pixels, exclude_current_fire)
    if mode != "test":
        if source_offset is None:
            raise ValueError("source_offset is required for merged train/val GOES spatial arrays.")
        goes = goes[source_offset:source_offset + masks.shape[0]]
    if goes.shape[0] != masks.shape[0]:
        raise ValueError(f"{location_id}: GOES rows {goes.shape[0]} do not match front masks {masks.shape[0]}")
    return goes[:] * masks[:, None, :, :, :]


def merge_chunk_files(mode: str, ts_length: int, interval: int, dtype: str) -> None:
    target_dir = DATASET_DIR / f"dataset_{mode}"
    paths = sorted(
        target_dir.glob(f"pred_{mode}_goes_spatial_frontbuf_seqtoseq_alll_{ts_length}i_{interval}_part_*_*.npy"),
        key=chunk_sort_key,
    )
    if not paths:
        raise FileNotFoundError(f"No GOES spatial front-buffer chunks found for {mode}.")

    arrays = [np.load(path, mmap_mode="r") for path in paths]
    shape = (sum(arr.shape[0] for arr in arrays), *arrays[0].shape[1:])
    out = np.lib.format.open_memmap(output_path(mode, ts_length, interval), mode="w+", dtype=np.dtype(dtype), shape=shape)
    offset = 0
    for arr in arrays:
        next_offset = offset + arr.shape[0]
        out[offset:next_offset] = arr.astype(dtype, copy=False)
        offset = next_offset
    out.flush()
    print(f"Merged {len(paths)} chunks into {output_path(mode, ts_length, interval)} with shape {shape}")


def generate_train_val(mode: str, locations: list[str], ts_length: int, interval: int, buffer_pixels: int, exclude_current_fire: bool, dtype: str) -> None:
    chunk_size = default_chunk_size(ts_length)
    offsets = cumulative_offsets(locations, ts_length, interval)
    for start in range(0, len(locations), chunk_size):
        chunk_locations = locations[start:start + chunk_size]
        end = start + len(chunk_locations)
        rows = []
        for location in tqdm(chunk_locations, desc=f"GOES front-buffer {mode} [{start}:{end}]", unit="fire"):
            arr = build_event_frontbuf(
                location,
                mode,
                ts_length,
                interval,
                buffer_pixels,
                exclude_current_fire,
                source_offset=offsets[location],
            )
            if arr.shape[0] > 0:
                rows.append(arr)
        if not rows:
            continue
        merged = np.concatenate(rows, axis=0).astype(dtype, copy=False)
        out = chunk_output_path(mode, ts_length, interval, start, end)
        np.save(out, merged)
        print(f"Wrote {merged.shape} to {out}")
    merge_chunk_files(mode, ts_length, interval, dtype)


def main() -> None:
    args = parse_args()
    if args.mode == "merge_train":
        merge_chunk_files("train", args.ts, args.it, args.dtype)
        return
    if args.mode == "merge_val":
        merge_chunk_files("val", args.ts, args.it, args.dtype)
        return

    locations = [location for location in resolve_locations(args.mode) if has_prediction_inputs(location)]
    locations = locations[max(args.start, 0):]
    if args.limit is not None:
        locations = locations[:args.limit]

    if args.mode == "test":
        for location in tqdm(locations, desc="GOES front-buffer test", unit="fire"):
            arr = build_event_frontbuf(location, args.mode, args.ts, args.it, args.buffer_pixels, args.exclude_current_fire)
            out = output_path(args.mode, args.ts, args.it, location)
            np.save(out, arr.astype(args.dtype, copy=False))
            print(f"{location}: wrote {arr.shape} to {out}")
        return

    if args.start > 0 or args.limit is not None:
        all_locations = [location for location in resolve_locations(args.mode) if has_prediction_inputs(location)]
        offsets = cumulative_offsets(all_locations, args.ts, args.it)
        rows = []
        for location in tqdm(locations, desc=f"GOES front-buffer {args.mode}", unit="fire"):
            arr = build_event_frontbuf(
                location,
                args.mode,
                args.ts,
                args.it,
                args.buffer_pixels,
                args.exclude_current_fire,
                source_offset=offsets[location],
            )
            if arr.shape[0] > 0:
                rows.append(arr)
        if not rows:
            raise RuntimeError("No front-buffer GOES rows generated.")
        end = args.start + len(locations)
        merged = np.concatenate(rows, axis=0).astype(args.dtype, copy=False)
        out = chunk_output_path(args.mode, args.ts, args.it, args.start, end)
        np.save(out, merged)
        print(f"Wrote {merged.shape} to {out}")
        return

    generate_train_val(args.mode, locations, args.ts, args.it, args.buffer_pixels, args.exclude_current_fire, args.dtype)


if __name__ == "__main__":
    main()
