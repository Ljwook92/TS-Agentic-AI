from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

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
            "Export per-fire VIIRS and GOES timestamp series as CSV and line charts "
            "to compare temporal resolution."
        )
    )
    parser.add_argument(
        "--viirs-root",
        default=str(get_raw_data_root()),
        help="Root directory of TS-SatFire raw fires.",
    )
    parser.add_argument(
        "--goes-root",
        required=True,
        help="Root directory of clipped GOES tif files.",
    )
    parser.add_argument(
        "--event-id",
        default="",
        help="Optional fire/event Id to export. If omitted, export all matching events.",
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=0,
        help="Optional maximum number of events to export. 0 means all.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "legacy" / "support" / "viirs_goes_timeseries"),
        help="Directory to write aggregate CSVs and per-fire PNG charts.",
    )
    parser.add_argument(
        "--goes-subdirs",
        default=",".join(DEFAULT_GOES_SUBDIRS),
        help="Comma-separated GOES leaf directory names to scan under each event.",
    )
    parser.add_argument(
        "--bin-hours",
        type=int,
        default=1,
        help="Hours per chart bin. Default: 1.",
    )
    return parser.parse_args()


def parse_timestamp_from_name(name: str) -> datetime | None:
    stem = Path(name).stem
    candidates: list[datetime] = []
    patterns = [
        (r"(?<!\d)(20\d{12})(?!\d)", "%Y%m%d%H%M%S"),
        (r"(?<!\d)(20\d{8})(?!\d)", "%Y%m%d%H"),
        (r"(?<!\d)(20\d{6})(?!\d)", "%Y%m%d"),
        (r"(?<!\d)(20\d{2})(\d{3})(\d{6})(?!\d)", None),
        (r"(?<!\d)s(20\d{2})(\d{3})(\d{6})(?!\d)", None),
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

    normalized = re.sub(r"[^0-9T]", "", stem)
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M", "%Y%m%d"):
        try:
            candidates.append(datetime.strptime(normalized, fmt))
        except ValueError:
            continue

    return min(candidates) if candidates else None


def dedupe_timed_paths(items: Iterable[TimedPath]) -> list[TimedPath]:
    seen: set[tuple[datetime, str]] = set()
    out: list[TimedPath] = []
    for item in items:
        key = (item.timestamp, item.path.name)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def iter_event_dirs(viirs_root: Path, event_id: str = "", limit_events: int = 0) -> list[Path]:
    if event_id:
        event_dir = viirs_root / event_id
        return [event_dir] if (event_dir / "VIIRS_Day").exists() else []
    event_dirs = sorted(path for path in viirs_root.iterdir() if (path / "VIIRS_Day").exists())
    if limit_events > 0:
        event_dirs = event_dirs[:limit_events]
    return event_dirs


def collect_viirs_times(event_dir: Path) -> list[TimedPath]:
    items: list[TimedPath] = []
    for path in sorted((event_dir / "VIIRS_Day").iterdir()):
        if path.suffix.lower() not in TIFF_SUFFIXES:
            continue
        timestamp = parse_timestamp_from_name(path.name)
        if timestamp is None:
            continue
        items.append(TimedPath(timestamp=timestamp, path=path))
    return dedupe_timed_paths(sorted(items, key=lambda item: (item.timestamp, item.path.name)))


def locate_goes_event_dirs(goes_root: Path, event_id: str, allowed_leaf_names: set[str]) -> list[Path]:
    matched: list[Path] = []
    for path in goes_root.rglob(event_id):
        if not path.is_dir():
            continue
        matched.extend(
            child for child in path.rglob("*")
            if child.is_dir() and child.name in allowed_leaf_names
        )
    if matched:
        return sorted(set(matched))
    return sorted(
        path for path in goes_root.rglob("*")
        if path.is_dir()
        and path.name in allowed_leaf_names
        and event_id in str(path.parent)
    )


def collect_goes_times(goes_root: Path, event_id: str, goes_subdirs: set[str]) -> dict[str, list[TimedPath]]:
    by_source: dict[str, list[TimedPath]] = {}
    for subdir in locate_goes_event_dirs(goes_root, event_id=event_id, allowed_leaf_names=goes_subdirs):
        source = subdir.name
        bucket = by_source.setdefault(source, [])
        for path in sorted(subdir.iterdir()):
            if path.suffix.lower() not in TIFF_SUFFIXES:
                continue
            timestamp = parse_timestamp_from_name(path.name)
            if timestamp is None:
                continue
            bucket.append(TimedPath(timestamp=timestamp, path=path))
    return {
        key: dedupe_timed_paths(sorted(value, key=lambda item: (item.timestamp, item.path.name)))
        for key, value in by_source.items()
    }


def write_raw_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event_id", "source", "timestamp", "filename", "path"])
        writer.writeheader()
        writer.writerows(rows)


def floor_to_bin(ts: datetime, bin_hours: int) -> datetime:
    floored_hour = (ts.hour // bin_hours) * bin_hours
    return ts.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def build_count_rows(event_id: str, series: dict[str, list[TimedPath]], bin_hours: int) -> list[dict[str, object]]:
    counts: dict[datetime, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sources = sorted(series)
    for source, items in series.items():
        for item in items:
            counts[floor_to_bin(item.timestamp, bin_hours)][source] += 1

    rows: list[dict[str, object]] = []
    for timestamp in sorted(counts):
        row: dict[str, object] = {
            "event_id": event_id,
            "bin_start": timestamp.isoformat(),
        }
        for source in sources:
            row[source] = counts[timestamp].get(source, 0)
        rows.append(row)
    return rows


def write_count_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_counts(path: Path, event_id: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    timestamps = [datetime.fromisoformat(str(row["bin_start"])) for row in rows]
    sources = [key for key in rows[0].keys() if key not in {"event_id", "bin_start"}]

    plt.figure(figsize=(12, 4))
    for source in sources:
        values = [int(row.get(source, 0)) for row in rows]
        plt.plot(timestamps, values, marker="o", linewidth=1.5, label=source)
    plt.title(f"{event_id} VIIRS vs GOES counts over time")
    plt.xlabel("Time")
    plt.ylabel("Files per bin")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def export_event(event_dir: Path, goes_root: Path, goes_subdirs: set[str], output_dir: Path, bin_hours: int) -> tuple[dict[str, object], list[dict[str, str]], list[dict[str, object]]]:
    event_id = event_dir.name
    viirs_times = collect_viirs_times(event_dir)
    goes_times = collect_goes_times(goes_root=goes_root, event_id=event_id, goes_subdirs=goes_subdirs)

    raw_rows: list[dict[str, str]] = []
    all_series: dict[str, list[TimedPath]] = {"viirs_day": viirs_times}
    all_series.update(goes_times)

    for source, items in all_series.items():
        for item in items:
            raw_rows.append(
                {
                    "event_id": event_id,
                    "source": source,
                    "timestamp": item.timestamp.isoformat(),
                    "filename": item.path.name,
                    "path": str(item.path),
                }
            )

    count_rows = build_count_rows(event_id=event_id, series=all_series, bin_hours=bin_hours)
    plots_dir = output_dir / "plots"
    plot_counts(plots_dir / f"{event_id}_counts_{bin_hours}h.png", event_id=event_id, rows=count_rows)

    summary = {
        "event_id": event_id,
        "viirs_count": len(viirs_times),
        "goes_counts": {key: len(value) for key, value in goes_times.items()},
        "plot_path": str(plots_dir / f"{event_id}_counts_{bin_hours}h.png"),
    }
    return summary, raw_rows, count_rows


def main() -> None:
    args = parse_args()
    viirs_root = Path(args.viirs_root)
    goes_root = Path(args.goes_root)
    output_dir = Path(args.output_dir)
    goes_subdirs = {item.strip() for item in args.goes_subdirs.split(",") if item.strip()}

    event_dirs = iter_event_dirs(viirs_root=viirs_root, event_id=args.event_id, limit_events=args.limit_events)
    if not event_dirs:
        raise SystemExit("No matching event directories with VIIRS_Day were found.")

    summaries = []
    all_raw_rows: list[dict[str, str]] = []
    all_count_rows: list[dict[str, object]] = []
    for event_dir in event_dirs:
        summary, raw_rows, count_rows = export_event(
            event_dir=event_dir,
            goes_root=goes_root,
            goes_subdirs=goes_subdirs,
            output_dir=output_dir,
            bin_hours=args.bin_hours,
        )
        summaries.append(summary)
        all_raw_rows.extend(raw_rows)
        all_count_rows.extend(count_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_raw_csv(output_dir / "all_raw_timestamps.csv", all_raw_rows)
    write_count_csv(output_dir / f"all_counts_{args.bin_hours}h.csv", all_count_rows)

    print(
        {
            "events_exported": len(summaries),
            "output_dir": str(output_dir),
            "bin_hours": args.bin_hours,
            "raw_csv": str(output_dir / "all_raw_timestamps.csv"),
            "count_csv": str(output_dir / f"all_counts_{args.bin_hours}h.csv"),
        }
    )
    for item in summaries[:10]:
        print(item)


if __name__ == "__main__":
    main()
