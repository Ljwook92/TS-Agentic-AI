from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.support.path_config import get_raw_data_root


TIFF_SUFFIXES = {".tif", ".tiff"}
DEFAULT_GOES_SUBDIRS = ("mask_fixed", "frp_fixed", "mask", "frp")


@dataclass(frozen=True)
class TimedPath:
    timestamp: datetime
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze temporal alignment between TS-SatFire VIIRS acquisitions and "
            "clipped GOES files for each fire event."
        )
    )
    parser.add_argument(
        "--viirs-root",
        default=str(get_raw_data_root()),
        help="Root directory of TS-SatFire raw fires. Defaults to SATFIRE_ROOT/ts-satfire.",
    )
    parser.add_argument(
        "--goes-root",
        required=True,
        help="Root directory of clipped GOES tif files.",
    )
    parser.add_argument(
        "--event-id",
        default="",
        help="Optional fire/event Id to analyze. If omitted, analyze all events with VIIRS_Day.",
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=0,
        help="Optional max number of events to analyze. 0 means all.",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Optional path to save per-VIIRS-interval alignment rows as CSV.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional path to save per-event summary JSON.",
    )
    parser.add_argument(
        "--goes-subdirs",
        default=",".join(DEFAULT_GOES_SUBDIRS),
        help="Comma-separated GOES leaf directory names to scan under each event.",
    )
    return parser.parse_args()


def iter_event_dirs(viirs_root: Path, event_id: str = "", limit_events: int = 0) -> list[Path]:
    if event_id:
        event_dir = viirs_root / event_id
        return [event_dir] if (event_dir / "VIIRS_Day").exists() else []

    event_dirs = sorted(path for path in viirs_root.iterdir() if (path / "VIIRS_Day").exists())
    if limit_events > 0:
        event_dirs = event_dirs[:limit_events]
    return event_dirs


def parse_timestamp_from_name(name: str) -> datetime | None:
    stem = Path(name).stem
    candidates: list[datetime] = []

    patterns = [
        (r"(?<!\d)(20\d{12})(?!\d)", "%Y%m%d%H%M%S"),
        (r"(?<!\d)(20\d{8})(?!\d)", "%Y%m%d%H"),
        (r"(?<!\d)(20\d{6})(?!\d)", "%Y%m%d"),
        (r"(?<!\d)(20\d{2})(\d{3})(\d{6})(?!\d)", None),  # YYYY DDD HHMMSS
        (r"(?<!\d)s(20\d{2})(\d{3})(\d{6})(?!\d)", None),  # GOES stem style
    ]

    for pattern, fmt in patterns:
        match = re.search(pattern, stem)
        if not match:
            continue
        if fmt is not None:
            try:
                candidates.append(datetime.strptime(match.group(1), fmt))
            except ValueError:
                continue
            continue

        year, doy, hms = match.groups()
        try:
            candidates.append(datetime.strptime(f"{year}{doy}{hms}", "%Y%j%H%M%S"))
        except ValueError:
            continue

    # Common formats with separators.
    normalized = re.sub(r"[^0-9T]", "", stem)
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M", "%Y%m%d"):
        try:
            parsed = datetime.strptime(normalized, fmt)
        except ValueError:
            continue
        candidates.append(parsed)

    if not candidates:
        return None
    return min(candidates)


def collect_viirs_times(event_dir: Path) -> list[TimedPath]:
    viirs_dir = event_dir / "VIIRS_Day"
    timed_paths: list[TimedPath] = []
    for path in sorted(viirs_dir.iterdir()):
        if path.suffix.lower() not in TIFF_SUFFIXES:
            continue
        timestamp = parse_timestamp_from_name(path.name)
        if timestamp is None:
            continue
        timed_paths.append(TimedPath(timestamp=timestamp, path=path))
    timed_paths.sort(key=lambda item: (item.timestamp, item.path.name))
    return dedupe_timed_paths(timed_paths)


def dedupe_timed_paths(items: Iterable[TimedPath]) -> list[TimedPath]:
    seen: set[tuple[datetime, str]] = set()
    deduped: list[TimedPath] = []
    for item in items:
        key = (item.timestamp, item.path.name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def locate_goes_event_dirs(goes_root: Path, event_id: str, allowed_leaf_names: set[str]) -> list[Path]:
    matched: list[Path] = []
    for path in goes_root.rglob(event_id):
        if not path.is_dir():
            continue
        matched.extend(
            child for child in path.rglob("*")
            if child.is_dir() and child.name in allowed_leaf_names
        )
    # Fall back to direct leaf-name scan when event folder names are not preserved.
    if matched:
        return sorted(set(matched))
    return sorted(
        path for path in goes_root.rglob("*")
        if path.is_dir()
        and path.name in allowed_leaf_names
        and event_id in str(path.parent)
    )


def collect_goes_times(goes_root: Path, event_id: str, goes_subdirs: set[str]) -> dict[str, list[TimedPath]]:
    timed_by_source: dict[str, list[TimedPath]] = {}
    for subdir in locate_goes_event_dirs(goes_root, event_id=event_id, allowed_leaf_names=goes_subdirs):
        source = subdir.name
        timed_paths = timed_by_source.setdefault(source, [])
        for path in sorted(subdir.iterdir()):
            if path.suffix.lower() not in TIFF_SUFFIXES:
                continue
            timestamp = parse_timestamp_from_name(path.name)
            if timestamp is None:
                continue
            timed_paths.append(TimedPath(timestamp=timestamp, path=path))
    return {key: dedupe_timed_paths(sorted(value, key=lambda item: (item.timestamp, item.path.name))) for key, value in timed_by_source.items()}


def slice_interval(items: list[TimedPath], start: datetime, end: datetime) -> list[TimedPath]:
    return [item for item in items if start < item.timestamp <= end]


def summarize_event(event_dir: Path, goes_root: Path, goes_subdirs: set[str]) -> tuple[list[dict[str, object]], dict[str, object]]:
    event_id = event_dir.name
    viirs_times = collect_viirs_times(event_dir)
    goes_by_source = collect_goes_times(goes_root=goes_root, event_id=event_id, goes_subdirs=goes_subdirs)

    interval_rows: list[dict[str, object]] = []
    if len(viirs_times) < 2:
        return interval_rows, {
            "event_id": event_id,
            "viirs_count": len(viirs_times),
            "interval_count": 0,
            "goes_sources": sorted(goes_by_source),
            "covered_intervals_any_goes": 0,
            "coverage_ratio_any_goes": 0.0,
        }

    covered_intervals = 0
    for idx in range(len(viirs_times) - 1):
        current_item = viirs_times[idx]
        next_item = viirs_times[idx + 1]
        interval_hours = (next_item.timestamp - current_item.timestamp).total_seconds() / 3600.0
        row: dict[str, object] = {
            "event_id": event_id,
            "interval_index": idx,
            "viirs_t": current_item.timestamp.isoformat(),
            "viirs_t_plus_1": next_item.timestamp.isoformat(),
            "viirs_gap_hours": round(interval_hours, 3),
        }
        has_any_goes = False
        for source, items in goes_by_source.items():
            interval_items = slice_interval(items, start=current_item.timestamp, end=next_item.timestamp)
            row[f"{source}_count"] = len(interval_items)
            row[f"{source}_first"] = interval_items[0].timestamp.isoformat() if interval_items else ""
            row[f"{source}_last"] = interval_items[-1].timestamp.isoformat() if interval_items else ""
            if interval_items:
                has_any_goes = True
        row["has_any_goes"] = int(has_any_goes)
        if has_any_goes:
            covered_intervals += 1
        interval_rows.append(row)

    summary = {
        "event_id": event_id,
        "viirs_count": len(viirs_times),
        "interval_count": len(interval_rows),
        "goes_sources": sorted(goes_by_source),
        "covered_intervals_any_goes": covered_intervals,
        "coverage_ratio_any_goes": round(covered_intervals / len(interval_rows), 4) if interval_rows else 0.0,
        "first_viirs_time": viirs_times[0].timestamp.isoformat(),
        "last_viirs_time": viirs_times[-1].timestamp.isoformat(),
    }
    return interval_rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    viirs_root = Path(args.viirs_root)
    goes_root = Path(args.goes_root)
    goes_subdirs = {item.strip() for item in args.goes_subdirs.split(",") if item.strip()}

    event_dirs = iter_event_dirs(viirs_root=viirs_root, event_id=args.event_id, limit_events=args.limit_events)
    if not event_dirs:
        raise SystemExit("No matching event directories with VIIRS_Day were found.")

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for event_dir in event_dirs:
        rows, summary = summarize_event(event_dir=event_dir, goes_root=goes_root, goes_subdirs=goes_subdirs)
        all_rows.extend(rows)
        summaries.append(summary)

    if args.output_csv and all_rows:
        write_csv(Path(args.output_csv), all_rows)
    if args.summary_json:
        write_json(Path(args.summary_json), summaries)

    print(
        json.dumps(
            {
                "events_analyzed": len(summaries),
                "events_with_viirs_intervals": sum(1 for item in summaries if item["interval_count"] > 0),
                "rows_written": len(all_rows),
                "output_csv": args.output_csv,
                "summary_json": args.summary_json,
            },
            indent=2,
        )
    )
    for item in summaries[:10]:
        print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()
