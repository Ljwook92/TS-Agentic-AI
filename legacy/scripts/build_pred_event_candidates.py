import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

try:
    from analyze_pred_event_windows import (
        count_new_adjacent,
        has_prediction_inputs,
        load_daily_masks,
        resolve_locations,
    )
except ModuleNotFoundError:
    from scripts.analyze_pred_event_windows import (
        count_new_adjacent,
        has_prediction_inputs,
        load_daily_masks,
        resolve_locations,
    )


DIRECTION_LABELS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def structure_for_connectivity(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError("connectivity must be 4 or 8")


def frontier_mask(component: np.ndarray, current_fire: np.ndarray, connectivity: int) -> np.ndarray:
    background = ~current_fire
    adjacent_to_background = ndimage.binary_dilation(
        background,
        structure=structure_for_connectivity(connectivity),
        iterations=1,
        border_value=1,
    )
    return component & adjacent_to_background


def direction_bin_8(row_delta: np.ndarray, col_delta: np.ndarray) -> np.ndarray:
    # 0=N, 1=NE, 2=E, ... with image rows increasing southward.
    degrees = (np.degrees(np.arctan2(col_delta, -row_delta)) + 360.0) % 360.0
    return np.floor((degrees + 22.5) / 45.0).astype(np.int16) % 8


def component_stats(labels: np.ndarray, n_components: int, current_fire: np.ndarray, connectivity: int) -> dict[int, dict]:
    stats: dict[int, dict] = {}
    for component_id in range(1, n_components + 1):
        component = labels == component_id
        area = int(component.sum())
        if area == 0:
            continue
        front = frontier_mask(component, current_fire, connectivity)
        coords = np.argwhere(component)
        centroid_row, centroid_col = coords.mean(axis=0)
        stats[component_id] = {
            "component_area": area,
            "component_front_pixels": int(front.sum()),
            "component_centroid_row": float(centroid_row),
            "component_centroid_col": float(centroid_col),
        }
    return stats


def iter_candidate_rows(
    fire_id: str,
    dates: list[str],
    masks: list[np.ndarray],
    connectivity: int,
    candidate_radius: float,
    min_component_pixels: int,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    summary_rows: list[dict] = []
    structure = structure_for_connectivity(connectivity)

    for day_idx in range(len(masks) - 1):
        current = masks[day_idx].astype(bool)
        future = masks[day_idx + 1].astype(bool)
        if not current.any():
            continue

        growth = future & ~current
        labels, n_components = ndimage.label(current, structure=structure)
        stats = component_stats(labels, n_components, current, connectivity)

        valid_component_ids = {
            component_id
            for component_id, item in stats.items()
            if item["component_area"] >= min_component_pixels
        }
        if not valid_component_ids:
            continue

        # For each non-fire candidate pixel, find the nearest existing fire pixel and its component.
        background = ~current
        distances, nearest_indices = ndimage.distance_transform_edt(
            background,
            return_indices=True,
        )
        nearest_rows = nearest_indices[0]
        nearest_cols = nearest_indices[1]
        nearest_component = labels[nearest_rows, nearest_cols]
        candidate_mask = background & (distances <= candidate_radius) & np.isin(
            nearest_component,
            list(valid_component_ids),
        )

        candidate_coords = np.argwhere(candidate_mask)
        if candidate_coords.size == 0:
            continue

        cand_rows = candidate_coords[:, 0]
        cand_cols = candidate_coords[:, 1]
        near_rows = nearest_rows[cand_rows, cand_cols]
        near_cols = nearest_cols[cand_rows, cand_cols]
        comp_ids = nearest_component[cand_rows, cand_cols].astype(np.int32)
        dists = distances[cand_rows, cand_cols]
        dir_bins = direction_bin_8(cand_rows - near_rows, cand_cols - near_cols)
        labels_next = growth[cand_rows, cand_cols].astype(np.int8)

        new_pixels, new_adjacent_pixels = count_new_adjacent(current, future, connectivity)
        positives_by_component: dict[int, int] = {}
        candidates_by_component: dict[int, int] = {}

        for i in range(candidate_coords.shape[0]):
            component_id = int(comp_ids[i])
            item = stats[component_id]
            label_next = int(labels_next[i])
            positives_by_component[component_id] = positives_by_component.get(component_id, 0) + label_next
            candidates_by_component[component_id] = candidates_by_component.get(component_id, 0) + 1
            rows.append({
                "fire_id": fire_id,
                "date": dates[day_idx],
                "next_date": dates[day_idx + 1],
                "day_idx": day_idx,
                "component_id": component_id,
                "candidate_row": int(cand_rows[i]),
                "candidate_col": int(cand_cols[i]),
                "nearest_fire_row": int(near_rows[i]),
                "nearest_fire_col": int(near_cols[i]),
                "distance_px": float(dists[i]),
                "direction_bin_8": int(dir_bins[i]),
                "direction_label_8": DIRECTION_LABELS_8[int(dir_bins[i])],
                "component_area": item["component_area"],
                "component_front_pixels": item["component_front_pixels"],
                "component_centroid_row": item["component_centroid_row"],
                "component_centroid_col": item["component_centroid_col"],
                "current_fire_pixels": int(current.sum()),
                "growth_pixels_next_day": new_pixels,
                "growth_adjacent_pixels_next_day": new_adjacent_pixels,
                "label_ignited_next_day": label_next,
            })

        for component_id in sorted(candidates_by_component):
            item = stats[component_id]
            summary_rows.append({
                "fire_id": fire_id,
                "date": dates[day_idx],
                "next_date": dates[day_idx + 1],
                "day_idx": day_idx,
                "component_id": component_id,
                "component_area": item["component_area"],
                "component_front_pixels": item["component_front_pixels"],
                "component_centroid_row": item["component_centroid_row"],
                "component_centroid_col": item["component_centroid_col"],
                "candidate_pixels": candidates_by_component[component_id],
                "positive_candidate_pixels": positives_by_component.get(component_id, 0),
            })

    return rows, summary_rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict], write_header: bool) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build component-wise candidate-grid wildfire spread rows for prediction events."
    )
    parser.add_argument("-mode", choices=["train", "val", "test"], default="test")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--candidate-radius", type=float, default=5.0)
    parser.add_argument("--min-component-pixels", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("output/event_candidates"))
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
    radius_tag = str(args.candidate_radius).replace(".", "p")
    suffix = f"{args.mode}_conn{args.connectivity}_r{radius_tag}_mincomp{args.min_component_pixels}"
    candidate_path = args.out_dir / f"pred_event_candidates_{suffix}.csv"
    summary_path = args.out_dir / f"pred_event_candidate_summary_{suffix}.csv"
    if candidate_path.exists():
        candidate_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    candidate_fields = [
        "fire_id", "date", "next_date", "day_idx", "component_id",
        "candidate_row", "candidate_col", "nearest_fire_row", "nearest_fire_col",
        "distance_px", "direction_bin_8", "direction_label_8",
        "component_area", "component_front_pixels", "component_centroid_row", "component_centroid_col",
        "current_fire_pixels", "growth_pixels_next_day", "growth_adjacent_pixels_next_day",
        "label_ignited_next_day",
    ]
    summary_fields = [
        "fire_id", "date", "next_date", "day_idx", "component_id",
        "component_area", "component_front_pixels", "component_centroid_row", "component_centroid_col",
        "candidate_pixels", "positive_candidate_pixels",
    ]

    total_candidates = 0
    total_positives = 0
    total_components = 0

    for idx, (fire_id, label_sel) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] {fire_id}")
        dates, masks = load_daily_masks(fire_id, label_sel)
        candidate_rows, summary_rows = iter_candidate_rows(
            fire_id=fire_id,
            dates=dates,
            masks=masks,
            connectivity=args.connectivity,
            candidate_radius=args.candidate_radius,
            min_component_pixels=args.min_component_pixels,
        )
        if candidate_rows:
            write_rows(candidate_path, candidate_fields, candidate_rows, write_header=total_candidates == 0)
            total_candidates += len(candidate_rows)
            total_positives += sum(int(row["label_ignited_next_day"]) for row in candidate_rows)
        if summary_rows:
            write_rows(summary_path, summary_fields, summary_rows, write_header=total_components == 0)
            total_components += len(summary_rows)

        print(
            f"  candidates={len(candidate_rows)} positives="
            f"{sum(int(row['label_ignited_next_day']) for row in candidate_rows)} components={len(summary_rows)}"
        )

    print(f"Wrote {candidate_path}")
    print(f"Wrote {summary_path}")
    print(
        "total_candidates={} total_positive_candidates={} positive_rate={:.6f} total_components={}".format(
            total_candidates,
            total_positives,
            total_positives / total_candidates if total_candidates else math.nan,
            total_components,
        )
    )


if __name__ == "__main__":
    main()
