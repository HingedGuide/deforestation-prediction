# preprocessing/sample_builder.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio

from deforestation_predictor.preprocessing.catalog import (
    get_records_for_dates,
    compute_variable_maxima,
)
from deforestation_predictor.preprocessing.windows import (
    get_input_window_range,
    CONTEXT_MONTHS,
    GAP_MONTHS,
)


def stack_rasters(records: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[pd.Timestamp]]:
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

    dates = list(records["date"].unique())
    variables = list(records["variable"].unique())

    per_time = []
    for d in dates:
        per_var = []
        for v in variables:
            row = records[(records["date"] == d) & (records["variable"] == v)]
            if row.empty:
                raise ValueError(f"Missing variable '{v}' at date {d} in stack_rasters.")
            path = row["path"].iloc[0]
            with rasterio.open(path) as src:
                arr = src.read(1)  # [H, W]
            per_var.append(arr)
        per_time.append(np.stack(per_var, axis=0))  # [V, H, W] at time d

    cube = np.stack(per_time, axis=1)  # [V, T, H, W]
    return cube, variables, dates


def normalize_cube_auto(
    X: np.ndarray,
    variables: List[str],
    *,
    catalog: pd.DataFrame | None = None,
    maxima: Dict[str, float] | None = None,
    nan_policy: str = "zero",    # "zero" or "nan"
    overflow: str = "clip",      # "clip" | "zero" | "ignore"
    cast: str | None = None,     # None | "uint8" | "float16"
    use_cache: bool = True,
) -> np.ndarray:
    """
    Normalize cube [V, T, H, W].

    You can either:
      - provide `maxima` directly, OR
      - provide a `catalog` so this function computes maxima itself
        (with optional caching by catalog id).

    Steps:
      1) get maxima per variable
      2) divide each channel by its max
      3) NaN/Inf handling
      4) overflow handling
      5) optional casting
    """
    if maxima is None:
        if catalog is None:
            raise ValueError("Either `maxima` or `catalog` must be provided.")

        # simple cache keyed by id(catalog)
        cache_key = id(catalog)
        cache_attr = "_max_cache"

        if use_cache and hasattr(normalize_cube_auto, cache_attr):
            cache: Dict[int, Dict[str, float]] = getattr(normalize_cube_auto, cache_attr)  # type: ignore[attr-defined]
        else:
            cache = {}

        if use_cache and cache_key in cache:
            maxima = cache[cache_key]
        else:
            maxima = compute_variable_maxima(catalog)
            if use_cache:
                cache[cache_key] = maxima
                setattr(normalize_cube_auto, cache_attr, cache)
    # now we definitely have a maxima dict
    Xn = X.astype(np.float32, copy=True)

    for i, var in enumerate(variables):
        m = maxima.get(var, 1.0)
        if not np.isfinite(m) or m == 0:
            m = 1.0

        arr = Xn[i] / m

        # NaN/Inf handling
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
        # "ignore" → do nothing

        Xn[i] = arr

    # casting
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


def build_sample(
    tile_id: str,
    target_date: datetime,
    catalog: pd.DataFrame,
    gt_catalog: pd.DataFrame,
    context: int = CONTEXT_MONTHS,
    gap: int = GAP_MONTHS,
    *,
    normalize: bool = True,
    overflow: str = "clip",
    nan_policy: str = "zero",
    cast: str | None = None,
    use_cache: bool = True,
    maxima: Dict[str, float] | None = None,
):
    """
    Build a single training sample for a tile and target date.

    Returns:
      X: np.ndarray [V, T, H, W]
      y: np.ndarray [H, W]
      meta: dict with tile_id, target_date, variables, dates
    """
    # 1) time window
    start, end = get_input_window_range(target_date, context=context, gap=gap)

    # 2) records subset
    subset = get_records_for_dates(catalog, tile_id, start, end)
    if subset.empty:
        raise ValueError(f"No input data for tile {tile_id} between {start} and {end}")

    # 3) stack into [V, T, H, W]
    X, variables, dates = stack_rasters(subset)

    # 4) normalize (using global helpers)
    if normalize:
        X = normalize_cube_auto(
            X,
            variables,
            catalog=catalog if maxima is None else None,
            maxima=maxima,
            nan_policy=nan_policy,
            overflow=overflow,
            cast=cast,
            use_cache=use_cache,
        )

    # 5) load GT
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
