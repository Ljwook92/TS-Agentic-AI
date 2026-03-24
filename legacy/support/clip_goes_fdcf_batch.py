import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.transform import Affine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clip all GOES-R FDCF files matching bbox row date ranges."
    )
    parser.add_argument("--goes-root", required=True, help="Root directory of downloaded GOES product.")
    parser.add_argument("--bbox-csv", required=True, help="Path to bbox CSV.")
    parser.add_argument("--output-root", required=True, help="Directory to store clipped outputs.")
    parser.add_argument(
        "--limit-events",
        type=int,
        default=0,
        help="Optional max number of events to process. 0 means all.",
    )
    parser.add_argument(
        "--limit-files-per-event",
        type=int,
        default=0,
        help="Optional max number of files per event. 0 means all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clipped files.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional path to save per-event clip summary.",
    )
    return parser.parse_args()


def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


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


def compute_xy_bounds(ds, row):
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
    return min(x_rad), max(x_rad), min(y_rad), max(y_rad)


def clip_dataset(ds, xy_bounds):
    x_min, x_max, y_min, y_max = xy_bounds
    x_slice = slice(x_min, x_max)
    if ds.y.values[0] > ds.y.values[-1]:
        y_slice = slice(y_max, y_min)
    else:
        y_slice = slice(y_min, y_max)
    return ds.sel(x=x_slice, y=y_slice)


def clear_netcdf_encoding(ds):
    ds = ds.copy()
    ds.encoding = {}
    for name in ds.coords:
        ds[name].encoding = {}
    for name in ds.data_vars:
        ds[name].encoding = {}
    return ds


def build_affine_transform(clipped):
    x = clipped.x.values
    y = clipped.y.values
    if x.size < 2 or y.size < 2:
        raise ValueError("Need at least 2 x/y coordinates to build a GeoTIFF transform.")

    x_res = float(x[1] - x[0])
    y_res = float(y[1] - y[0])
    x_origin = float(x[0] - (x_res / 2.0))
    y_origin = float(y[0] - (y_res / 2.0))
    return Affine(x_res, 0.0, x_origin, 0.0, y_res, y_origin)


def extract_timestamp(nc_path, clipped):
    time_str = clipped.attrs.get("time_coverage_start", "")
    if time_str:
        return (
            time_str.replace("-", "")
            .replace(":", "")
            .replace(".0Z", "Z")
            .replace(".9Z", "Z")
            .replace(".5Z", "Z")
            .replace(".6Z", "Z")
            .replace(".7Z", "Z")
            .replace(".8Z", "Z")
            .replace("Z", "")
        )
    stem = nc_path.stem
    if "_s" in stem:
        return stem.split("_s", 1)[1].split("_", 1)[0]
    return stem


def write_geotiff(output_path, array, crs_wkt, transform, nodata):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs_wkt,
        transform=transform,
        nodata=nodata,
        compress="LZW",
    ) as dst:
        dst.write(array, 1)


def save_mask_and_frp(nc_path, clipped, output_root, fire_id, overwrite):
    crs_wkt = build_goes_crs(clipped).to_wkt()
    transform = build_affine_transform(clipped)
    timestamp = extract_timestamp(nc_path, clipped)

    mask_dir = output_root / fire_id / "mask"
    frp_dir = output_root / fire_id / "frp"
    mask_path = mask_dir / f"{timestamp}_mask.tif"
    frp_path = frp_dir / f"{timestamp}_frp.tif"

    if mask_path.exists() and frp_path.exists() and not overwrite:
        return "exists"

    mask_fill = clipped["Mask"].attrs.get("_FillValue", -99)
    mask_array = clipped["Mask"].values
    if np.issubdtype(mask_array.dtype, np.floating):
        mask_array = np.where(np.isnan(mask_array), mask_fill, mask_array)
    mask_array = mask_array.astype(np.int16, copy=False)

    power_fill = clipped["Power"].attrs.get("_FillValue", -9999.0)
    power_array = clipped["Power"].values
    power_array = np.where(np.isnan(power_array), power_fill, power_array).astype(np.float32, copy=False)

    write_geotiff(mask_path, mask_array, crs_wkt, transform, mask_fill)
    write_geotiff(frp_path, power_array, crs_wkt, transform, float(power_fill))
    return "saved"


def list_event_files(goes_root, start_date, end_date):
    files = []
    for current_date in daterange(start_date, end_date):
        year = f"{current_date.year:04d}"
        doy = f"{current_date.timetuple().tm_yday:03d}"
        day_dir = goes_root / year / doy
        if day_dir.exists():
            files.extend(sorted(day_dir.rglob("*.nc")))
    return files


def clip_one_file(nc_path, row, output_root, overwrite):
    fire_id = row["Id"]
    ds = xr.open_dataset(nc_path)
    xy_bounds = compute_xy_bounds(ds, row)
    clipped = clip_dataset(ds, xy_bounds)

    if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
        ds.close()
        return "empty"

    clipped = clear_netcdf_encoding(clipped)
    result = save_mask_and_frp(nc_path, clipped, output_root, fire_id, overwrite)
    ds.close()
    return result


def main():
    args = parse_args()
    goes_root = Path(args.goes_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.bbox_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if args.limit_events > 0:
        rows = rows[: args.limit_events]

    summary_rows = []

    for idx, row in enumerate(rows, start=1):
        fire_id = row["Id"]
        start_date = dt.date.fromisoformat(row["start_date"])
        end_date = dt.date.fromisoformat(row["end_date"])
        event_files = list_event_files(goes_root, start_date, end_date)

        if args.limit_files_per_event > 0:
            event_files = event_files[: args.limit_files_per_event]

        saved_count = 0
        empty_count = 0
        exists_count = 0
        error_count = 0

        print(f"[EVENT {idx}/{len(rows)}] {fire_id} files={len(event_files)}")

        for nc_path in event_files:
            try:
                result = clip_one_file(nc_path, row, output_root, args.overwrite)
            except Exception as exc:
                error_count += 1
                print(f"[ERROR] {fire_id} {nc_path}: {exc}")
                continue

            if result == "saved":
                saved_count += 1
            elif result == "empty":
                empty_count += 1
            elif result == "exists":
                exists_count += 1

        summary_rows.append(
            {
                "Id": fire_id,
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "input_files": len(event_files),
                "saved_files": saved_count,
                "empty_files": empty_count,
                "existing_files": exists_count,
                "error_files": error_count,
            }
        )

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Id",
                    "start_date",
                    "end_date",
                    "input_files",
                    "saved_files",
                    "empty_files",
                    "existing_files",
                    "error_files",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved clip summary to {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
