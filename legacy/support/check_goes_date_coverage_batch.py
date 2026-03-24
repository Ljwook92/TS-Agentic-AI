import argparse
import csv
import datetime as dt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check GOES date coverage for every row in a bbox CSV and save a summary CSV."
    )
    parser.add_argument("--goes-root", required=True, help="Root directory of downloaded GOES product.")
    parser.add_argument("--bbox-csv", required=True, help="Path to bbox CSV.")
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to save per-event coverage summary CSV.",
    )
    return parser.parse_args()


def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def collect_day_counts(goes_root, start_date, end_date):
    day_counts = []
    total_files = 0

    for current_date in daterange(start_date, end_date):
        year = f"{current_date.year:04d}"
        doy = f"{current_date.timetuple().tm_yday:03d}"
        day_dir = goes_root / year / doy

        if day_dir.exists():
            count = sum(1 for _ in day_dir.rglob("*.nc"))
        else:
            count = 0

        total_files += count
        day_counts.append((current_date.isoformat(), count))

    return day_counts, total_files


def main():
    args = parse_args()
    goes_root = Path(args.goes_root)

    with open(args.bbox_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {args.bbox_csv}")

    summary_rows = []

    for row in rows:
        fire_id = row.get("Id", "")
        start_raw = row.get("start_date", "")
        end_raw = row.get("end_date", "")
        if not start_raw or not end_raw:
            summary_rows.append(
                {
                    "Id": fire_id,
                    "start_date": start_raw,
                    "end_date": end_raw,
                    "total_days": 0,
                    "covered_days": 0,
                    "missing_days": 0,
                    "coverage_ratio": 0.0,
                    "total_files": 0,
                    "missing_date_list": "",
                }
            )
            continue

        start_date = dt.date.fromisoformat(start_raw)
        end_date = dt.date.fromisoformat(end_raw)
        day_counts, total_files = collect_day_counts(goes_root, start_date, end_date)

        total_days = len(day_counts)
        covered_days = sum(1 for _, count in day_counts if count > 0)
        missing_dates = [date_str for date_str, count in day_counts if count == 0]
        missing_days = len(missing_dates)
        coverage_ratio = covered_days / total_days if total_days else 0.0

        summary_rows.append(
            {
                "Id": fire_id,
                "start_date": start_raw,
                "end_date": end_raw,
                "total_days": total_days,
                "covered_days": covered_days,
                "missing_days": missing_days,
                "coverage_ratio": f"{coverage_ratio:.4f}",
                "total_files": total_files,
                "missing_date_list": ";".join(missing_dates),
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Id",
                "start_date",
                "end_date",
                "total_days",
                "covered_days",
                "missing_days",
                "coverage_ratio",
                "total_files",
                "missing_date_list",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    total_events = len(summary_rows)
    full_coverage_events = sum(1 for row in summary_rows if int(row["missing_days"]) == 0)
    partial_coverage_events = total_events - full_coverage_events

    print("Saved summary CSV:", output_path)
    print(
        {
            "total_events": total_events,
            "full_coverage_events": full_coverage_events,
            "partial_or_missing_events": partial_coverage_events,
        }
    )
    print("\nFirst 10 summary rows:")
    for row in summary_rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
