import argparse
import csv
import datetime as dt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether downloaded GOES files cover the date range for one bbox row."
    )
    parser.add_argument("--goes-root", required=True, help="Root directory of downloaded GOES product.")
    parser.add_argument("--bbox-csv", required=True, help="Path to bbox CSV.")
    parser.add_argument("--id", default="", help="Fire/event Id to select from bbox CSV.")
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="0-based row index to use when --id is not provided.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max number of matching file paths to print.",
    )
    return parser.parse_args()


def load_bbox_row(csv_path, fire_id, row_index):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    if fire_id:
        for row in rows:
            if row.get("Id", "") == fire_id:
                return row
        raise ValueError(f"Id {fire_id} not found in {csv_path}")

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index {row_index} out of range for {csv_path}")
    return rows[row_index]


def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def main():
    args = parse_args()
    row = load_bbox_row(args.bbox_csv, args.id, args.row_index)

    fire_id = row.get("Id", "")
    start_date = dt.date.fromisoformat(row["start_date"])
    end_date = dt.date.fromisoformat(row["end_date"])
    goes_root = Path(args.goes_root)

    print("Selected bbox row:")
    print(
        {
            "Id": fire_id,
            "start_date": row["start_date"],
            "end_date": row["end_date"],
        }
    )

    all_matches = []
    day_summaries = []

    for current_date in daterange(start_date, end_date):
        year = f"{current_date.year:04d}"
        doy = f"{current_date.timetuple().tm_yday:03d}"
        day_dir = goes_root / year / doy

        if not day_dir.exists():
            day_summaries.append((current_date.isoformat(), 0))
            continue

        day_files = sorted(day_dir.rglob("*.nc"))
        day_summaries.append((current_date.isoformat(), len(day_files)))
        all_matches.extend(day_files)

    covered_days = sum(1 for _, count in day_summaries if count > 0)
    total_days = len(day_summaries)

    print("\nCoverage summary:")
    print(
        {
            "total_days_in_range": total_days,
            "days_with_files": covered_days,
            "days_missing_files": total_days - covered_days,
            "total_matching_files": len(all_matches),
        }
    )

    print("\nPer-day file counts:")
    for date_str, count in day_summaries:
        print(f"{date_str}: {count}")

    if all_matches:
        print(f"\nFirst {min(args.limit, len(all_matches))} matching files:")
        for path in all_matches[: args.limit]:
            print(path)
    else:
        print("\nNo matching files found for this date range.")


if __name__ == "__main__":
    main()
