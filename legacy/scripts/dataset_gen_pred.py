import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from satimg_dataset_processor.satimg_dataset_processor import PredDatasetProcessor
from support.path_config import get_code_root, get_raw_data_root, get_task_dataset_root

RAW_DATA_DIR = str(get_raw_data_root())
DATASET_DIR = Path(get_task_dataset_root("pred"))
ROI_DIR = get_code_root() / "legacy" / "roi"


def has_prediction_inputs(location_id: str) -> bool:
    location_root = os.path.join(RAW_DATA_DIR, location_id)
    viirs_day_dir = os.path.join(location_root, "VIIRS_Day")
    firepred_dir = os.path.join(location_root, "FirePred")
    if not all(os.path.isdir(path) for path in (viirs_day_dir, firepred_dir)):
        return False

    viirs_day_files = [name for name in os.listdir(viirs_day_dir) if name.endswith(".tif")]
    firepred_files = [name for name in os.listdir(firepred_dir) if name.endswith(".tif")]
    return bool(viirs_day_files) and bool(firepred_files)


def load_roi_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_years = []
    for year in ["2017", "2018", "2019", "2020"]:
        filename = ROI_DIR / f"us_fire_{year}_out_new.csv"
        train_years.append(pd.read_csv(filename))

    test_years = []
    for year in ["2021"]:
        filename = ROI_DIR / f"us_fire_{year}_out_new.csv"
        test_years.append(pd.read_csv(filename))

    return pd.concat(train_years, ignore_index=True), pd.concat(test_years, ignore_index=True)


def canonical_paths(mode: str, ts_length: int, interval: int) -> tuple[Path, Path]:
    target_dir = DATASET_DIR / f"dataset_{mode}"
    target_dir.mkdir(parents=True, exist_ok=True)
    image_path = target_dir / f"pred_{mode}_img_seqtoseq_alll_{ts_length}i_{interval}.npy"
    label_path = target_dir / f"pred_{mode}_label_seqtoseq_alll_{ts_length}i_{interval}.npy"
    return image_path, label_path


def chunk_paths(mode: str, ts_length: int, interval: int, start: int, end: int) -> tuple[Path, Path]:
    target_dir = DATASET_DIR / f"dataset_{mode}"
    target_dir.mkdir(parents=True, exist_ok=True)
    image_path = target_dir / f"pred_{mode}_img_seqtoseq_alll_{ts_length}i_{interval}_part_{start}_{end}.npy"
    label_path = target_dir / f"pred_{mode}_label_seqtoseq_alll_{ts_length}i_{interval}_part_{start}_{end}.npy"
    return image_path, label_path


def merge_chunk_files(mode: str, ts_length: int, interval: int) -> None:
    if mode not in {"train", "val"}:
        raise ValueError("Chunk merge is only supported for train/val.")

    target_dir = DATASET_DIR / f"dataset_{mode}"
    img_chunks = sorted(target_dir.glob(f"pred_{mode}_img_seqtoseq_alll_{ts_length}i_{interval}_part_*_*.npy"))
    label_chunks = sorted(target_dir.glob(f"pred_{mode}_label_seqtoseq_alll_{ts_length}i_{interval}_part_*_*.npy"))

    if not img_chunks or not label_chunks:
        raise FileNotFoundError(f"No chunk files found for mode={mode}, ts={ts_length}, interval={interval}.")
    if len(img_chunks) != len(label_chunks):
        raise RuntimeError("Image chunk count and label chunk count do not match.")

    img_arrays = [np.load(path, mmap_mode="r") for path in img_chunks]
    label_arrays = [np.load(path, mmap_mode="r") for path in label_chunks]

    total_rows = sum(arr.shape[0] for arr in img_arrays)
    img_shape = (total_rows, *img_arrays[0].shape[1:])
    label_shape = (sum(arr.shape[0] for arr in label_arrays), *label_arrays[0].shape[1:])

    image_path, label_path = canonical_paths(mode=mode, ts_length=ts_length, interval=interval)
    if image_path.exists():
        image_path.unlink()
    if label_path.exists():
        label_path.unlink()

    img_out = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.float32, shape=img_shape)
    label_out = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.float32, shape=label_shape)

    img_offset = 0
    for arr in img_arrays:
        next_offset = img_offset + arr.shape[0]
        img_out[img_offset:next_offset] = arr
        img_offset = next_offset

    label_offset = 0
    for arr in label_arrays:
        next_offset = label_offset + arr.shape[0]
        label_out[label_offset:next_offset] = arr
        label_offset = next_offset

    img_out.flush()
    label_out.flush()
    print(f"Merged {len(img_chunks)} chunk files into {image_path} and {label_path}")


def resolve_locations(mode: str) -> tuple[list[str], list[int] | None]:
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
        ids = train_split["Id"].tolist()
        return ids, None
    if mode == "val":
        ids = val_split["Id"].tolist()
        return ids, None

    ids = test_df["Id"].tolist()
    labels = test_df["label_sel"].astype(int).tolist()
    return ids, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prediction datasets.")
    parser.add_argument("-mode", type=str, choices=["train", "val", "test", "merge_train", "merge_val"])
    parser.add_argument("-ts", type=int, help="Length of TS")
    parser.add_argument("-it", type=int, help="Interval")
    parser.add_argument("-limit", type=int, default=None, help="Optional limit on number of fires to process")
    parser.add_argument("-start", type=int, default=0, help="Optional start index into the filtered fire list")
    args = parser.parse_args()

    ts_length = args.ts
    interval = args.it
    mode = args.mode

    if mode == "merge_train":
        merge_chunk_files("train", ts_length=ts_length, interval=interval)
        raise SystemExit(0)
    if mode == "merge_val":
        merge_chunk_files("val", ts_length=ts_length, interval=interval)
        raise SystemExit(0)

    locations, test_label_sels = resolve_locations(mode)
    filtered_pairs: list[tuple[str, int | None]] = []
    for idx, location in enumerate(locations):
        if not has_prediction_inputs(location):
            continue
        label_sel = None if test_label_sels is None else test_label_sels[idx]
        filtered_pairs.append((location, label_sel))

    start = max(args.start, 0)
    filtered_pairs = filtered_pairs[start:]
    if args.limit is not None:
        filtered_pairs = filtered_pairs[:args.limit]

    if not filtered_pairs:
        raise RuntimeError(f"No valid prediction inputs found for mode={mode} after filtering.")

    satimg_processor = PredDatasetProcessor()

    if mode in ["train", "val"]:
        selected_locations = [location for location, _ in filtered_pairs]
        if args.start > 0 or args.limit is not None:
            end = start + len(selected_locations)
            image_path, label_path = chunk_paths(mode=mode, ts_length=ts_length, interval=interval, start=start, end=end)
        else:
            image_path, label_path = canonical_paths(mode=mode, ts_length=ts_length, interval=interval)

        satimg_processor.pred_dataset_generator_seqtoseq(
            mode=mode,
            locations=selected_locations,
            visualize=False,
            data_path=RAW_DATA_DIR,
            file_name=image_path.name,
            label_name=label_path.name,
            save_path=str(image_path.parent),
            ts_length=ts_length,
            interval=interval,
            image_size=(256, 256),
        )
        print(f"Wrote {len(selected_locations)} locations for mode={mode} to {image_path} and {label_path}")
    else:
        target_dir = DATASET_DIR / "dataset_test"
        target_dir.mkdir(parents=True, exist_ok=True)
        for location, label_sel in filtered_pairs:
            print(location)
            satimg_processor.pred_dataset_generator_seqtoseq(
                mode="test",
                locations=[location],
                visualize=False,
                data_path=RAW_DATA_DIR,
                file_name=f"pred_{location}_img_seqtoseql_{ts_length}i_{interval}.npy",
                label_name=f"pred_{location}_label_seqtoseql_{ts_length}i_{interval}.npy",
                save_path=str(target_dir),
                ts_length=ts_length,
                interval=interval,
                rs_idx=0.3,
                cs_idx=0.3,
                image_size=(256, 256),
                label_sel=0 if label_sel is None else label_sel,
            )
        print(f"Wrote {len(filtered_pairs)} test fires into {target_dir}")
