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
    maxima = {}

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


def normalize_cube_auto(
    X: np.ndarray,
    variables: list[str],
    catalog: pd.DataFrame,
    nan_policy: str = "zero",
    overflow: str = "clip",
    cast: str | None = None,
) -> np.ndarray:
    """
    Normalize cube [V, T, H, W] using per-variable maxima detected from catalog.
    Steps:
      1) compute maxima per variable
      2) divide each channel by its max
      3) handle NaN/Inf
      4) handle overflow
      5) optional cast (uint8 uses rounding)
    """
    maxima = compute_variable_maxima(catalog)

    # work in float32 first
    Xn = X.astype(np.float32, copy=True)

    for i, var in enumerate(variables):
        m = maxima.get(var, 1.0)
        if not np.isfinite(m) or m == 0:
            m = 1.0

        arr = Xn[i] / m

        # NaN/Inf -> 0 (or keep, if requested)
        if nan_policy == "zero":
            arr = np.where(np.isfinite(arr), arr, 0.0)

        # overflow handling
        if overflow == "clip":
            arr = np.clip(arr, 0.0, 1.0)
        elif overflow == "zero":
            over = arr > 1.0
            if np.any(over):
                arr = arr.copy()
                arr[over] = 0.0
        # "ignore": do nothing

        Xn[i] = arr

    # casting happens AFTER normalization
    if cast == "uint8":
        Xn = np.clip(Xn, 0.0, 1.0)
        Xn = np.rint(Xn * 255.0).astype(np.uint8, copy=False)
    elif cast == "float16":
        Xn = Xn.astype(np.float16, copy=False)
    elif cast in (None, "float32"):
        pass
    else:
        raise ValueError(f"Unsupported cast={cast}")

    return Xn


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
    *,
    normalize: bool = True,
    overflow: str = "clip",      # "clip" (safe) or "zero" (set >1 to 0) or "ignore"
    nan_policy: str = "zero",    # "zero" or "nan"
    cast: str | None = None,     # None, "uint8", "float16"
    use_cache: bool = True,
    maxima: dict[str, float] | None = None,  # pass to override/catalog-scan
):
    """
    Build a single training sample for a tile and target date.
    Adds automatic per-variable normalization based on maxima detected in the catalog.

    Returns:
      X: np.ndarray [V, T, H, W]   (optionally normalized/cast)
      y: np.ndarray [H, W]
      meta: dict (variables, dates, tile_id, target_date)
    """
    # ---- helpers ----
    def compute_variable_maxima(catalog_df: pd.DataFrame) -> dict[str, float]:
        """Scan all rasters per variable and return {var: global_max} (NaNs ignored)."""
        maxima_local: dict[str, float] = {}
        for var in catalog_df["variable"].unique():
            max_val = -np.inf
            var_paths = catalog_df.loc[catalog_df["variable"] == var, "path"].tolist()
            for p in var_paths:
                try:
                    with rasterio.open(p) as src:
                        arr = src.read(1).astype(np.float32)
                        arr = np.where(np.isfinite(arr), arr, np.nan)
                        local_max = np.nanmax(arr)
                        if np.isfinite(local_max) and local_max > max_val:
                            max_val = float(local_max)
                except Exception as e:
                    print(f"[Warning] Could not read {p}: {e}")
            maxima_local[var] = max_val if np.isfinite(max_val) else 1.0
        return maxima_local

    def normalize_cube_with_maxima(
        X: np.ndarray, variables: list[str], maxima_map: dict[str, float]
    ) -> np.ndarray:
        """Divide each [i, :, :, :] by maxima_map[variables[i]], handle NaN/Inf, overflow, and cast."""
        Xn = X.astype(np.float32, copy=True)

        for i, var in enumerate(variables):
            m = maxima_map.get(var, 1.0)
            if not np.isfinite(m) or m == 0:
                m = 1.0
            arr = Xn[i] / m if normalize else Xn[i]

            # NaN/Inf handling
            if nan_policy == "zero":
                arr = np.where(np.isfinite(arr), arr, 0.0)

            # Overflow handling
            if overflow == "clip":
                arr = np.clip(arr, 0.0, 1.0)
            elif overflow == "zero":
                over = arr > 1.0
                if np.any(over):
                    arr = arr.copy()
                    arr[over] = 0.0
            # "ignore": do nothing

            Xn[i] = arr

        # Casting
        if cast == "uint8":
            Xn = np.clip(Xn, 0.0, 1.0)
            Xn = (Xn * 255.0).astype(np.uint8, copy=False)
        elif cast == "float16":
            Xn = Xn.astype(np.float16, copy=False)
        elif cast in (None, "float32"):
            pass
        else:
            raise ValueError(f"Unsupported cast={cast}")

        return Xn

    # ---- 1) determine input time window ----
    start, end = get_input_window_range(target_date, context=context, gap=gap)

    # ---- 2) get all records for this tile and time window ----
    subset = get_records_for_dates(catalog, tile_id, start, end)
    if subset.empty:
        raise ValueError(f"No input data for tile {tile_id} between {start} and {end}")

    # ---- 3) stack into [V, T, H, W] ----
    X, variables, dates = stack_rasters(subset)

    # ---- 4) normalize using auto-detected maxima (cached) ----
    maxima_map: dict[str, float]
    if maxima is not None:
        maxima_map = maxima
    else:
        # lightweight cache keyed by object id of the catalog
        cache_key = (id(catalog),)
        if use_cache and hasattr(build_sample, "_max_cache"):
            maxima_cache: dict[tuple[int], dict[str, float]] = build_sample._max_cache  # type: ignore[attr-defined]
        else:
            maxima_cache = {}

        if use_cache and cache_key in maxima_cache:
            maxima_map = maxima_cache[cache_key]
        else:
            maxima_map = compute_variable_maxima(catalog)
            if use_cache:
                maxima_cache[cache_key] = maxima_map
                build_sample._max_cache = maxima_cache  # type: ignore[attr-defined]

    X = normalize_cube_with_maxima(X, variables, maxima_map)

    # ---- 5) load GT for this tile and target_date ----
    T = pd.to_datetime(target_date)
    gt_rows = gt_catalog[(gt_catalog["tile_id"] == tile_id) & (gt_catalog["date"] == T)]
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

