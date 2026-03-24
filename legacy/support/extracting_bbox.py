import argparse
import csv
import json
import math
import os


def km_to_lat_deg(km):
    return km / 110.574


def km_to_lon_deg(km, lat):
    cos_lat = math.cos(math.radians(lat))
    if abs(cos_lat) < 1e-8:
        raise ValueError(f"Longitude conversion is unstable near the poles for latitude={lat}")
    return km / (111.320 * cos_lat)


def build_bbox(lat, lon, half_height_km, half_width_km):
    lat_offset = km_to_lat_deg(half_height_km)
    lon_offset = km_to_lon_deg(half_width_km, lat)
    return {
        "min_lon": lon - lon_offset,
        "min_lat": lat - lat_offset,
        "max_lon": lon + lon_offset,
        "max_lat": lat + lat_offset,
    }


def make_feature(row, bbox):
    coordinates = [[
        [bbox["min_lon"], bbox["min_lat"]],
        [bbox["min_lon"], bbox["max_lat"]],
        [bbox["max_lon"], bbox["max_lat"]],
        [bbox["max_lon"], bbox["min_lat"]],
        [bbox["min_lon"], bbox["min_lat"]],
    ]]
    properties = dict(row)
    properties.update(bbox)
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create WGS84 bounding boxes from ROI center coordinates."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="roi/fire_all.csv",
        help="Input CSV containing center coordinates.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="roi/fire_all_bbox.csv",
        help="Output CSV path for per-event bbox rows.",
    )
    parser.add_argument(
        "--geojson",
        default="",
        help="Optional GeoJSON output path for bbox polygons.",
    )
    parser.add_argument(
        "--lat-col",
        default="lat",
        help="Latitude column name.",
    )
    parser.add_argument(
        "--lon-col",
        default="lon",
        help="Longitude column name.",
    )
    parser.add_argument(
        "--id-col",
        default="Id",
        help="Identifier column name.",
    )
    parser.add_argument(
        "--start-date-col",
        default="start_date",
        help="Start date column name.",
    )
    parser.add_argument(
        "--end-date-col",
        default="end_date",
        help="End date column name.",
    )
    parser.add_argument(
        "--half-size-km",
        type=float,
        default=128.0,
        help="Half-size of a square bbox in kilometers.",
    )
    parser.add_argument(
        "--half-width-km",
        type=float,
        default=None,
        help="Optional half-width in kilometers. Overrides --half-size-km for width.",
    )
    parser.add_argument(
        "--half-height-km",
        type=float,
        default=None,
        help="Optional half-height in kilometers. Overrides --half-size-km for height.",
    )
    parser.add_argument(
        "--aggregate-output",
        default="",
        help="Optional CSV path for a single bbox covering all rows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    half_width_km = args.half_width_km if args.half_width_km is not None else args.half_size_km
    half_height_km = args.half_height_km if args.half_height_km is not None else args.half_size_km

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    output_rows = []
    features = []
    aggregate = {
        "min_lon": float("inf"),
        "min_lat": float("inf"),
        "max_lon": float("-inf"),
        "max_lat": float("-inf"),
    }
    start_dates = []
    end_dates = []

    for row in rows:
        lat = float(row[args.lat_col])
        lon = float(row[args.lon_col])
        bbox = build_bbox(lat, lon, half_height_km, half_width_km)
        start_date = row.get(args.start_date_col, "")
        end_date = row.get(args.end_date_col, "")

        aggregate["min_lon"] = min(aggregate["min_lon"], bbox["min_lon"])
        aggregate["min_lat"] = min(aggregate["min_lat"], bbox["min_lat"])
        aggregate["max_lon"] = max(aggregate["max_lon"], bbox["max_lon"])
        aggregate["max_lat"] = max(aggregate["max_lat"], bbox["max_lat"])
        if start_date:
            start_dates.append(start_date)
        if end_date:
            end_dates.append(end_date)

        output_row = {
            args.id_col: row.get(args.id_col, ""),
            args.start_date_col: start_date,
            args.end_date_col: end_date,
            "center_lat": lat,
            "center_lon": lon,
            "half_width_km": half_width_km,
            "half_height_km": half_height_km,
            "min_lon": bbox["min_lon"],
            "min_lat": bbox["min_lat"],
            "max_lon": bbox["max_lon"],
            "max_lat": bbox["max_lat"],
        }
        for key, value in row.items():
            if key not in output_row:
                output_row[key] = value
        output_rows.append(output_row)
        features.append(make_feature(row, output_row))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = list(output_rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    if args.geojson:
        geojson_dir = os.path.dirname(args.geojson)
        if geojson_dir:
            os.makedirs(geojson_dir, exist_ok=True)
        with open(args.geojson, "w") as f:
            json.dump({
                "type": "FeatureCollection",
                "features": features,
            }, f, indent=2)

    if args.aggregate_output:
        aggregate_dir = os.path.dirname(args.aggregate_output)
        if aggregate_dir:
            os.makedirs(aggregate_dir, exist_ok=True)
        with open(args.aggregate_output, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_file",
                    "row_count",
                    "start_date_min",
                    "end_date_max",
                    "half_width_km",
                    "half_height_km",
                    "min_lon",
                    "min_lat",
                    "max_lon",
                    "max_lat",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "source_file": args.input,
                "row_count": len(output_rows),
                "start_date_min": min(start_dates) if start_dates else "",
                "end_date_max": max(end_dates) if end_dates else "",
                "half_width_km": half_width_km,
                "half_height_km": half_height_km,
                "min_lon": aggregate["min_lon"],
                "min_lat": aggregate["min_lat"],
                "max_lon": aggregate["max_lon"],
                "max_lat": aggregate["max_lat"],
            })

    print(f"Saved per-event bbox CSV to {args.output}")
    if args.geojson:
        print(f"Saved bbox GeoJSON to {args.geojson}")
    if args.aggregate_output:
        print(f"Saved aggregate bbox CSV to {args.aggregate_output}")


if __name__ == "__main__":
    main()
