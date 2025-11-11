from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pandas import DateOffset

CONTEXT_MONTHS = 12
GAP_MONTHS = 3


def parse_filename(filename: str | Path) -> dict:
    """
    Parse a filename formatted as:
        'coordinate1_coordinate2_YYYY-MM-DD_variable.tif'

    Returns a dictionary with:
        - tile_id (str)
        - date (datetime)
        - variable (str)
        - path (str)
    """
    filename = Path(filename)
    name = filename.stem  # remove '.tif'
    parts = name.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename.name}")

    coord1 = parts[0]
    coord2 = parts[1]
    date_str = parts[2]
    variable = "_".join(parts[3:])  # supports variables with underscores

    tile_id = f"{coord1}_{coord2}"
    date = datetime.strptime(date_str, "%Y-%m-%d")

    return {
        "tile_id": tile_id,
        "date": date,
        "variable": variable,
        "path": str(filename),
    }


def build_raster_catalog(data_root: str) -> pd.DataFrame:
    """
    Scan data_root recursively for .tif files and parse their metadata
    into a structured DataFrame using parse_filename().

    Columns:
        - tile_id
        - date
        - variable
        - path
    """
    paths = Path(data_root).rglob("*.tif")
    records = [parse_filename(p) for p in paths]

    df = pd.DataFrame(records)
    df.sort_values(["tile_id", "date", "variable"], inplace=True)
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


def stack_rasters(records: pd.DataFrame):
    """
    Stack rasters from a catalog subset into an array [V, T, H, W].

    Assumes:
      - records contains only one tile_id
      - records has columns: ['date', 'variable', 'path']
      - all rasters share the same spatial grid (H, W)

    Returns:
      cube: np.ndarray [V, T, H, W]
      variables: list[str]
      dates: list[pd.Timestamp]
    """
    if records.empty:
        raise ValueError("No records provided to stack_rasters.")

    # records are assumed sorted by date, variable already
    dates = list(records["date"].unique())
    variables = list(records["variable"].unique())

    per_time = []

    for d in dates:
        per_var = []
        for v in variables:
            row = records[(records["date"] == d) & (records["variable"] == v)]
            if row.empty:
                raise ValueError(
                    f"Missing variable '{v}' at date {d} in stack_rasters."
                )
            path = row["path"].iloc[0]
            with rasterio.open(path) as src:
                arr = src.read(1)  # [H, W]
            per_var.append(arr)
        # [V, H, W] for this time step
        per_time.append(np.stack(per_var, axis=0))

    # [V, T, H, W]
    cube = np.stack(per_time, axis=1)

    return cube, variables, dates


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


def get_input_window_range(
    target_date: datetime | pd.Timestamp,
    context: int = CONTEXT_MONTHS,
    gap: int = GAP_MONTHS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Given a target_date T, return (start, end) for the input window.

    For context=12, gap=3:
      start = T - 14 months
      end   = T - 3 months
    """
    T = pd.to_datetime(target_date)
    start = T - DateOffset(months=context + gap - 1)  # T - 14
    end = T - DateOffset(months=gap)                  # T - 3
    return start, end


def build_sample(
    tile_id: str,
    target_date: datetime,
    catalog: pd.DataFrame,
    gt_catalog: pd.DataFrame,
    context: int = CONTEXT_MONTHS,
    gap: int = GAP_MONTHS,
):
    """
    Build a single training sample for a tile and target date.

    Returns:
      X: np.ndarray of shape [V, T, H, W]
      y: np.ndarray of shape [H, W]
      meta: dict with info (variables, dates, tile_id, target_date)
    """
    # 1) determine input time window
    start, end = get_input_window_range(target_date, context=context, gap=gap)

    # 2) get all records for this tile and time window
    subset = get_records_for_dates(catalog, tile_id, start, end)
    if subset.empty:
        raise ValueError(
            f"No input data for tile {tile_id} between {start} and {end}"
        )

    # 3) stack into [V, T, H, W]
    X, variables, dates = stack_rasters(subset)

    # 4) load GT for this tile and target_date
    T = pd.to_datetime(target_date)
    gt_rows = gt_catalog[
        (gt_catalog["tile_id"] == tile_id) & (gt_catalog["date"] == T)
    ]

    if gt_rows.empty:
        raise ValueError(f"No GT found for tile {tile_id} and date {T}")

    gt_path = gt_rows["path"].iloc[0]
    with rasterio.open(gt_path) as src:
        y = src.read(1)  # [H, W]

    meta = {
        "tile_id": tile_id,
        "target_date": T,
        "variables": variables,
        "dates": dates,
    }

    return X, y, meta
