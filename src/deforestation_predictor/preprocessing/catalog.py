from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio

from deforestation_predictor.utils.filenames import parse_filename


# Optional: list of variables you consider temporally aggregated and want to drop
AGGREGATED_VARS: set[str] = {
    "lastsixmonths",
    "lastthreemonths",
    "lastmonth",
    "smoothedsixmonths",
    "smoothedtotal",
    "previoussameseason",
    "patchdensity",
    "totallossalerts",
    # extend with any others you know are aggregated
}


def build_raster_catalog(
    data_root: str,
    *,
    allowed_variables: list[str] | None = None,
    drop_aggregated: bool = False,
) -> pd.DataFrame:
    """
    Scan data_root recursively for .tif files and parse their metadata
    into a structured DataFrame using parse_filename().

    Columns:
        - tile_id
        - date
        - variable
        - path

    Parameters
    ----------
    data_root : str
        Root folder containing input rasters.
    allowed_variables : list[str] | None
        If provided, only keep rows where variable is in this list.
        This is the safest way to ensure you only use snapshot variables.
    drop_aggregated : bool
        If True (and allowed_variables is None), drop variables listed
        in AGGREGATED_VARS.
    """
    paths = Path(data_root).rglob("*.tif")
    records = [parse_filename(p) for p in paths]

    df = pd.DataFrame(records)

    if allowed_variables is not None:
        df = df[df["variable"].isin(allowed_variables)].copy()
    elif drop_aggregated:
        df = df[~df["variable"].isin(AGGREGATED_VARS)].copy()

    df.sort_values(["tile_id", "date", "variable"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_gt_catalog(gt_root: str) -> pd.DataFrame:
    """
    Scan gt_root for .tif files and build a GT catalog with:
        - tile_id
        - date
        - variable
        - path

    Only keeps files where variable == 'gt' (case-insensitive).
    """
    paths = Path(gt_root).rglob("*.tif")
    records = []
    for p in paths:
        info = parse_filename(p)
        if info["variable"].lower() != "gt":
            continue
        records.append(
            {
                "tile_id": info["tile_id"],
                "date": info["date"],
                "variable": info["variable"],
                "path": info["path"],
            }
        )

    df = pd.DataFrame(records)
    df.sort_values(["tile_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_records_for_dates(
    catalog: pd.DataFrame,
    tile_id: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Retrieve records from the catalog for a specific tile_id
    within a date range [start, end] (inclusive).

    Returns a DataFrame sorted by ['date', 'variable'].
    """
    subset = catalog[
        (catalog["tile_id"] == tile_id)
        & (catalog["date"] >= start)
        & (catalog["date"] <= end)
    ].copy()

    subset.sort_values(["date", "variable"], inplace=True)
    subset.reset_index(drop=True, inplace=True)
    return subset


def compute_variable_maxima(catalog: pd.DataFrame) -> dict[str, float]:
    """
    Compute per-variable maximum values across all tiles and dates.

    Reads each raster once (band 1) and returns a dictionary:
        { variable_name: max_value }

    NaNs are ignored. Inf values are treated as NaN.

    Note:
        - This can be slow for large datasets because it reads all rasters.
          You can sample by tile/date if needed.
    """
    maxima: dict[str, float] = {}

    for var in catalog["variable"].unique():
        var_paths = catalog.loc[catalog["variable"] == var, "path"].tolist()
        max_val = -np.inf

        for p in var_paths:
            try:
                with rasterio.open(p) as src:
                    arr = src.read(1).astype(np.float32)
                    arr = np.where(np.isfinite(arr), arr, np.nan)
                    local_max = np.nanmax(arr)
                    if local_max > max_val:
                        max_val = local_max
            except Exception as e:
                print(f"[Warning] Could not read {p}: {e}")

        # fallback if all were nan or unreadable
        if not np.isfinite(max_val):
            max_val = 1.0

        maxima[var] = float(max_val)

    return maxima
