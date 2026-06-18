import argparse
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from satimg_dataset_processor.utils import SatProcessingUtils
from support.path_config import get_code_root, get_raw_data_root


RAW_DATA_DIR = Path(get_raw_data_root())
ROI_DIR = get_code_root() / "legacy" / "roi"

OUTPUT_SIZE = 256
OFFSET = 128


def load_roi_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_years = []
    for year in ["2017", "2018", "2019", "2020"]:
        train_years.append(pd.read_csv(ROI_DIR / f"us_fire_{year}_out_new.csv"))

    test_df = pd.read_csv(ROI_DIR / "us_fire_2021_out_new.csv")
    return pd.concat(train_years, ignore_index=True), test_df


def resolve_locations(mode: str) -> list[tuple[str, int]]:
    train_df, test_df = load_roi_tables()
    val_ids = {
        "20568194", "20701026", "20562846", "20700973", "24462610", "24462788", "24462753",
        "24103571", "21998313", "21751303", "22141596", "21999381", "23301962", "22712904", "22713339",
    }

    train_df = train_df.sort_values(by=["Id"]).copy()
    train_df["Id"] = train_df["Id"].astype(str)
    train_df["label_sel"] = 1

    test_df = test_df.sort_values(by=["Id"]).copy()
    test_df["Id"] = test_df["Id"].astype(str)
    if "label_sel" not in test_df:
        test_df["label_sel"] = 1

    if mode == "train":
        df = train_df[~train_df.Id.isin(val_ids)]
    elif mode == "val":
        df = train_df[train_df.Id.isin(val_ids)]
    elif mode == "test":
        df = test_df
    elif mode == "all":
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return [(str(row.Id), int(row.label_sel)) for row in df.itertuples()]


def has_prediction_inputs(location_id: str) -> bool:
    location_root = RAW_DATA_DIR / location_id
    return (location_root / "VIIRS_Day").is_dir() and (location_root / "FirePred").is_dir()


def parse_date(path: Path) -> str:
    stem = path.stem
    match = re.search(r"(\d{4}[-_]?\d{2}[-_]?\d{2})", stem)
    if match:
        value = match.group(1).replace("_", "-")
        if "-" not in value:
            value = f"{value[:4]}-{value[4:6]}-{value[6:8]}"
        return value
    return stem.replace("_VIIRS_Day", "")


def crop_bool(arr: np.ndarray) -> np.ndarray:
    cropped = arr[OFFSET:OFFSET + OUTPUT_SIZE, OFFSET:OFFSET + OUTPUT_SIZE]
    return np.nan_to_num(cropped, nan=0.0, posinf=0.0, neginf=0.0) > 0


def neighbor_mask(mask: np.ndarray, connectivity: int) -> np.ndarray:
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")

    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in offsets:
        out |= padded[1 + dr:1 + dr + mask.shape[0], 1 + dc:1 + dc + mask.shape[1]]
    return out


def load_daily_masks(location_id: str, label_sel: int) -> tuple[list[str], list[np.ndarray]]:
    day_dir = RAW_DATA_DIR / location_id / "VIIRS_Day"
    files = sorted(day_dir.glob("*.tif"))
    if not files:
        return [], []

    reader = SatProcessingUtils()
    cumulative_af = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)
    cumulative_ba = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)
    dates: list[str] = []
    masks: list[np.ndarray] = []

    for file in files:
        arr, _ = reader.read_tiff(str(file))
        if arr.shape[0] >= 8:
            ba = crop_bool(arr[7])
        else:
            ba = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)
        af = crop_bool(arr[6]) if arr.shape[0] >= 7 else np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)

        cumulative_af |= af
        cumulative_ba |= ba
        if label_sel == 1:
            final_mask = cumulative_af.copy()
        else:
            final_mask = (cumulative_af | cumulative_ba).copy()

        dates.append(parse_date(file))
        masks.append(final_mask)

    return dates, masks


def count_new_adjacent(current: np.ndarray, future: np.ndarray, connectivity: int) -> tuple[int, int]:
    growth = future & ~current
    adjacent_growth = growth & neighbor_mask(current, connectivity)
    return int(growth.sum()), int(adjacent_growth.sum())


def sequential_events(
    location_id: str,
    dates: list[str],
    masks: list[np.ndarray],
    connectivity: int,
    min_new_pixels: int,
) -> list[dict]:
    events = []
    if not masks:
        return events

    start = next((idx for idx, mask in enumerate(masks) if mask.any()), None)
    if start is None:
        return events

    while start < len(masks) - 1:
        current = masks[start]
        found = False
        for end in range(start + 1, len(masks)):
            new_pixels, new_adjacent_pixels = count_new_adjacent(current, masks[end], connectivity)
            if new_adjacent_pixels >= min_new_pixels:
                events.append({
                    "fire_id": location_id,
                    "start_idx": start,
                    "end_idx": end,
                    "start_date": dates[start],
                    "end_date": dates[end],
                    "wait_days": end - start,
                    "start_pixels": int(current.sum()),
                    "end_pixels": int(masks[end].sum()),
                    "new_pixels": new_pixels,
                    "new_adjacent_pixels": new_adjacent_pixels,
                })
                start = end
                found = True
                break
        if not found:
            events.append({
                "fire_id": location_id,
                "start_idx": start,
                "end_idx": None,
                "start_date": dates[start],
                "end_date": None,
                "wait_days": None,
                "start_pixels": int(current.sum()),
                "end_pixels": None,
                "new_pixels": 0,
                "new_adjacent_pixels": 0,
            })
            break

    return events


def start_day_gaps(
    location_id: str,
    dates: list[str],
    masks: list[np.ndarray],
    connectivity: int,
    min_new_pixels: int,
) -> list[dict]:
    rows = []
    for start, current in enumerate(masks[:-1]):
        if not current.any():
            continue
        matched = None
        matched_counts = (0, 0)
        for end in range(start + 1, len(masks)):
            counts = count_new_adjacent(current, masks[end], connectivity)
            if counts[1] >= min_new_pixels:
                matched = end
                matched_counts = counts
                break
        rows.append({
            "fire_id": location_id,
            "start_idx": start,
            "end_idx": matched,
            "start_date": dates[start],
            "end_date": None if matched is None else dates[matched],
            "wait_days": None if matched is None else matched - start,
            "start_pixels": int(current.sum()),
            "new_pixels": matched_counts[0],
            "new_adjacent_pixels": matched_counts[1],
        })
    return rows


def summarize_fire(events: list[dict], location_id: str, n_days: int) -> dict:
    completed = [event for event in events if event["wait_days"] is not None]
    waits = [int(event["wait_days"]) for event in completed]
    counter = Counter(waits)
    return {
        "fire_id": location_id,
        "n_days": n_days,
        "n_events": len(completed),
        "n_open_events": len(events) - len(completed),
        "mean_wait_days": float(np.mean(waits)) if waits else np.nan,
        "median_wait_days": float(np.median(waits)) if waits else np.nan,
        "min_wait_days": int(np.min(waits)) if waits else None,
        "max_wait_days": int(np.max(waits)) if waits else None,
        "wait_1d": counter.get(1, 0),
        "wait_2d": counter.get(2, 0),
        "wait_3d": counter.get(3, 0),
        "wait_4_7d": sum(value for key, value in counter.items() if 4 <= key <= 7),
        "wait_gt7d": sum(value for key, value in counter.items() if key > 7),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize variable-horizon event windows for TS-SatFire prediction fires."
    )
    parser.add_argument("-mode", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--min-new-pixels", type=int, default=1)
    parser.add_argument("--event-policy", choices=["sequential", "start_day", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("output/event_windows"))
    parser.add_argument("-limit", type=int, default=None)
    parser.add_argument("-start", type=int, default=0)
    args = parser.parse_args()

    pairs = [(fire_id, label_sel) for fire_id, label_sel in resolve_locations(args.mode) if has_prediction_inputs(fire_id)]
    pairs = pairs[max(args.start, 0):]
    if args.limit is not None:
        pairs = pairs[:args.limit]
    if not pairs:
        raise RuntimeError(f"No valid fires found for mode={args.mode}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sequential_rows: list[dict] = []
    start_day_rows: list[dict] = []
    summary_rows: list[dict] = []

    for idx, (fire_id, label_sel) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] {fire_id}")
        dates, masks = load_daily_masks(fire_id, label_sel)
        events = sequential_events(
            location_id=fire_id,
            dates=dates,
            masks=masks,
            connectivity=args.connectivity,
            min_new_pixels=args.min_new_pixels,
        )
        summary = summarize_fire(events, fire_id, len(dates))
        summary["label_sel"] = label_sel
        summary_rows.append(summary)

        if args.event_policy in {"sequential", "both"}:
            sequential_rows.extend(events)
        if args.event_policy in {"start_day", "both"}:
            start_day_rows.extend(start_day_gaps(
                location_id=fire_id,
                dates=dates,
                masks=masks,
                connectivity=args.connectivity,
                min_new_pixels=args.min_new_pixels,
            ))

    suffix = f"{args.mode}_conn{args.connectivity}_min{args.min_new_pixels}"
    summary_path = args.out_dir / f"event_window_summary_{suffix}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    if sequential_rows:
        sequential_path = args.out_dir / f"event_windows_sequential_{suffix}.csv"
        pd.DataFrame(sequential_rows).to_csv(sequential_path, index=False)
        print(f"Wrote {sequential_path}")

    if start_day_rows:
        start_day_path = args.out_dir / f"event_windows_start_day_{suffix}.csv"
        pd.DataFrame(start_day_rows).to_csv(start_day_path, index=False)
        print(f"Wrote {start_day_path}")

    summary_df = pd.DataFrame(summary_rows)
    total_events = int(summary_df["n_events"].sum()) if not summary_df.empty else 0
    print(f"fires={len(summary_df)} total_completed_events={total_events}")
    if total_events:
        event_df = pd.DataFrame(sequential_rows)
        waits = event_df["wait_days"].dropna().astype(int)
        print("sequential wait_days distribution:")
        print(waits.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
