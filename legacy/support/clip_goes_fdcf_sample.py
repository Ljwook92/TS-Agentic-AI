import argparse
import csv
import os

import xarray as xr
from pyproj import CRS, Transformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clip one GOES-R FDCF NetCDF file using one bbox row from a CSV."
    )
    parser.add_argument("--nc", required=True, help="Path to one GOES-R FDCF NetCDF file.")
    parser.add_argument("--bbox-csv", required=True, help="Path to bbox CSV.")
    parser.add_argument("--id", default="", help="Fire/event Id to select from bbox CSV.")
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="0-based row index to use when --id is not provided.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output NetCDF path for the clipped subset.",
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


def build_goes_crs(ds):
    proj = ds["goes_imager_projection"]
    return CRS.from_proj4(
        " ".join(
            [
                "+proj=geos",
                f"+h={proj.perspective_point_height.item()}",
                f"+lon_0={proj.longitude_of_projection_origin.item()}",
                f"+sweep={proj.sweep_angle_axis}",
                "+ellps=GRS80",
                "+no_defs",
            ]
        )
    )


def clip_dataset(ds, row):
    min_lon = float(row["min_lon"])
    min_lat = float(row["min_lat"])
    max_lon = float(row["max_lon"])
    max_lat = float(row["max_lat"])

    proj = ds["goes_imager_projection"]
    sat_height = proj.perspective_point_height.item()

    goes_crs = build_goes_crs(ds)
    transformer = Transformer.from_crs("EPSG:4326", goes_crs, always_xy=True)

    corner_lons = [min_lon, min_lon, max_lon, max_lon]
    corner_lats = [min_lat, max_lat, min_lat, max_lat]
    x_m, y_m = transformer.transform(corner_lons, corner_lats)

    x_rad = [value / sat_height for value in x_m]
    y_rad = [value / sat_height for value in y_m]

    x_min = min(x_rad)
    x_max = max(x_rad)
    y_min = min(y_rad)
    y_max = max(y_rad)

    x_slice = slice(x_min, x_max)
    if ds.y.values[0] > ds.y.values[-1]:
        y_slice = slice(y_max, y_min)
    else:
        y_slice = slice(y_min, y_max)

    clipped = ds.sel(x=x_slice, y=y_slice)
    return clipped, (x_min, x_max, y_min, y_max)


def summarize(clipped):
    print(clipped)
    print("\nData variables:", list(clipped.data_vars))
    for name in ["Mask", "DQF", "Area", "Temp", "Power"]:
        if name in clipped:
            data = clipped[name]
            print(f"{name} shape: {tuple(data.shape)} dtype: {data.dtype}")
            try:
                print(f"{name} min/max: {float(data.min())} / {float(data.max())}")
            except Exception:
                pass


def main():
    args = parse_args()
    row = load_bbox_row(args.bbox_csv, args.id, args.row_index)

    print("Selected bbox row:")
    print(
        {
            "Id": row.get("Id", ""),
            "start_date": row.get("start_date", ""),
            "end_date": row.get("end_date", ""),
            "min_lon": row.get("min_lon", ""),
            "min_lat": row.get("min_lat", ""),
            "max_lon": row.get("max_lon", ""),
            "max_lat": row.get("max_lat", ""),
        }
    )

    ds = xr.open_dataset(args.nc)
    clipped, bounds = clip_dataset(ds, row)
    print("\nProjected bounds in GOES x/y radians:")
    print(
        {
            "x_min": bounds[0],
            "x_max": bounds[1],
            "y_min": bounds[2],
            "y_max": bounds[3],
        }
    )

    if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
        raise RuntimeError("Clip returned an empty subset. Check satellite/region overlap.")

    print("\nClipped subset summary:")
    summarize(clipped)

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        clipped.to_netcdf(args.output)
        print(f"\nSaved clipped subset to {args.output}")


if __name__ == "__main__":
    main()
