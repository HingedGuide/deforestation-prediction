from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.windows import from_bounds, transform as window_transform

from deforestation_predictor.preprocessing.catalog import (
    build_raster_catalog,
    build_gt_catalog,
)


Bounds = Tuple[float, float, float, float]  # (west, south, east, north)


def _mosaic_and_clip_group(
    paths: list[Path],
    bounds: Bounds,
    out_path: Path,
) -> None:
    """
    Mosaic a list of single-band rasters and clip to the given bounding box.

    Parameters
    ----------
    paths : list[Path]
        Paths to rasters of the same variable/date, but different tiles.
    bounds : (west, south, east, north)
        Bounding box in the same CRS as the rasters (likely EPSG:4326).
    out_path : Path
        Where to write the output clipped mosaic.
    """
    if not paths:
        return

    src_files = [rasterio.open(p) for p in paths]

    try:
        # Mosaic → [bands, H, W], transform
        mosaic_arr, mosaic_transform = rio_merge(src_files)

        # Clip to bounds
        west, south, east, north = bounds
        window = from_bounds(
            west, south, east, north,
            transform=mosaic_transform,
        )

        # window.height/width may be floats; cast to int for slicing
        row_off = int(window.row_off)
        col_off = int(window.col_off)
        height = int(window.height)
        width = int(window.width)

        clipped = mosaic_arr[
            :,
            row_off : row_off + height,
            col_off : col_off + width,
        ]
        clipped_transform = window_transform(window, mosaic_transform)

        # Use meta from first source as template
        src0 = src_files[0]
        meta = src0.meta.copy()
        meta.update(
            {
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": clipped_transform,
                "count": clipped.shape[0],  # usually 1
            }
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(clipped)

    finally:
        for s in src_files:
            s.close()


def mosaic_and_clip_region(
    *,
    raw_input_root: Path,
    raw_gt_root: Path,
    out_input_root: Path,
    out_gt_root: Path,
    region_id: str,
    bounds: Bounds,
    tile_ids: Iterable[str],
) -> None:
    """
    Build mosaiced + clipped rasters for a region from several tiles.

    This will:
      - Read all input variables from `raw_input_root`
      - Read all ground-truth rasters from `raw_gt_root`
      - Filter to the given tile_ids
      - Mosaic & clip per (date, variable)
      - Write new rasters with tile_id = region_id

    Output filenames have the pattern:
        {region_id}_{YYYY-MM-DD}_{variable}.tif
    so that `parse_filename` still works.
    """
    tile_ids = set(tile_ids)

    # ---------- INPUT VARIABLES ----------
    print(f"[0] Building mosaiced INPUT for region {region_id}...")
    in_cat = build_raster_catalog(str(raw_input_root))
    if in_cat.empty:
        print("    [warning] No input rasters found, skipping inputs.")
    else:
        in_cat = in_cat[in_cat["tile_id"].isin(tile_ids)].copy()
        print(f"    -> {len(in_cat)} input rasters across "
              f"{in_cat['tile_id'].nunique()} tiles")

        # Group by (date, variable) → mosaic each group
        for (date, var), group in in_cat.groupby(["date", "variable"]):
            paths = [Path(p) for p in group["path"].tolist()]
            out_path = (
                out_input_root
                / region_id
                / f"{region_id}_{date.date()}_{var}.tif"
            )
            _mosaic_and_clip_group(paths, bounds, out_path)

    # ---------- GROUND TRUTH ----------
    print(f"[0] Building mosaiced GROUNDTRUTH for region {region_id}...")
    gt_cat = build_gt_catalog(str(raw_gt_root))
    if gt_cat.empty:
        print("    [warning] No GT rasters found, skipping GT.")
    else:
        gt_cat = gt_cat[gt_cat["tile_id"].isin(tile_ids)].copy()
        print(f"    -> {len(gt_cat)} GT rasters across "
              f"{gt_cat['tile_id'].nunique()} tiles")

        for (date, var), group in gt_cat.groupby(["date", "variable"]):
            paths = [Path(p) for p in group["path"].tolist()]
            out_path = (
                out_gt_root
                / region_id
                / f"{region_id}_{date.date()}_{var}.tif"
            )
            _mosaic_and_clip_group(paths, bounds, out_path)
