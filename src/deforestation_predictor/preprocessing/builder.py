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
    skip_vars: set[str] | None = None,
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
        arr = Xn[i]

        # 1) scale by maxima unless this var is in skip_vars
        if skip_vars is None or var not in skip_vars:
            m = maxima.get(var, 1.0)
            if not np.isfinite(m) or m == 0:
                m = 1.0
            arr = arr / m

        # 2) NaN / Inf handling
        if nan_policy == "zero":
            arr = np.where(np.isfinite(arr), arr, 0.0)

        # 3) Overflow handling
        if overflow == "clip":
            arr = np.clip(arr, 0.0, 1.0)
        elif overflow == "zero":
            over = arr > 1.0
            if np.any(over):
                arr = arr.copy()
                arr[over] = 0.0
        # "ignore" → do nothing

        Xn[i] = arr

    # casting (unchanged)
    if cast == "uint8":
        Xn = np.clip(Xn, 0.0, 1.0)
        Xn = np.rint(Xn * 255.0).astype(np.uint8, copy=False)
    elif cast == "float16":
        Xn = Xn.astype(np.float16, copy=False)

    return Xn


def build_sample(
    tile_id: str,
    target_date: datetime,
    catalog: pd.DataFrame,
    gt_catalog: pd.DataFrame,
    context: int = CONTEXT_MONTHS,
    gap: int = GAP_MONTHS,
    *,
    static_catalog: pd.DataFrame | None = None,
    normalize: bool = True,
    overflow: str = "clip",
    nan_policy: str = "zero",
    cast: str | None = None,
    use_cache: bool = True,
    maxima: Dict[str, float] | None = None,
    forestmask_catalog: pd.DataFrame | None = None,
    ignore_label: int = 2,
    forest_mask_threshold: float = 2000.0,
    categorical_vars: set[str] | None = None,
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
            skip_vars=categorical_vars,
        )
        if static_catalog is not None and not static_catalog.empty:
            static_records = static_catalog[static_catalog["tile_id"] == tile_id]

            if not static_records.empty:
                static_vars = sorted(static_records["variable"].unique())
                static_arrays = []

                for v in static_vars:
                    # some static vars have multiple years; take the most recent snapshot
                    rows_v = static_records[static_records["variable"] == v].sort_values("date")
                    path = rows_v["path"].iloc[-1]
                    with rasterio.open(path) as src:
                        arr = src.read(1).astype(np.float32)  # [H, W]
                    static_arrays.append(arr)

                static_cube = np.stack(static_arrays, axis=0)  # [V_static, H, W]

                # normalize static vars using same maxima dict
                if normalize:
                    static_cube_4d = static_cube[:, None, :, :]  # [V_static, 1, H, W]
                    static_cube_4d = normalize_cube_auto(
                        static_cube_4d,
                        static_vars,
                        catalog=None,
                        maxima=maxima,
                        nan_policy=nan_policy,
                        overflow=overflow,
                        cast=None,  # keep full precision here; casting happens later if you want
                        use_cache=False,
                        skip_vars=categorical_vars,
                    )
                    static_cube = static_cube_4d[:, 0, :, :]  # back to [V_static, H, W]

                # broadcast over time dimension
                V_dyn, T_len, H, W = X.shape
                static_broadcast = np.repeat(static_cube[:, None, :, :], T_len, axis=1)  # [V_static, T, H, W]

                # concatenate along channel dimension
                X = np.concatenate([X, static_broadcast], axis=0)
                variables = list(variables) + static_vars

    # 5) load GT
    T = pd.to_datetime(target_date)
    gt_rows = gt_catalog[(gt_catalog["tile_id"] == tile_id) & (gt_catalog["date"] == T)]
    if gt_rows.empty:
        raise ValueError(f"No GT found for tile {tile_id} and date {T}")

    gt_path = gt_rows["path"].iloc[0]
    with rasterio.open(gt_path) as src:
        y = src.read(1).astype(np.float32)  # [H, W]

    # ---- Make target binary: deforested (1) vs not-deforested (0) ----
    # Any positive value is treated as "deforestation"
    y_bin = np.zeros_like(y, dtype=np.uint8)
    y_bin[y > 0] = 1

    # ---- Apply forest mask from initialforestcover, if available ----
    if forestmask_catalog is not None and not forestmask_catalog.empty:
        mask_rows = forestmask_catalog[forestmask_catalog["tile_id"] == tile_id]

        if not mask_rows.empty:
            # Take most recent snapshot if there are multiple dates
            mask_rows = mask_rows.sort_values("date")
            mask_path = mask_rows["path"].iloc[-1]

            with rasterio.open(mask_path) as src:
                forest_raw = src.read(1).astype(np.float32)

            # Pixels with value > mask_threshold are treated as forest
            forest_mask = forest_raw > forest_mask_threshold

            # Everything outside forest becomes IGNORE_LABEL
            y_bin[~forest_mask] = ignore_label

            # Optional: track this in meta
            # (we'll add meta a few lines later)
        else:
            # Optional: print a warning if you expect a mask for every tile
            print(f"[Warning] No forest mask found for tile {tile_id}")

    meta = {
        "tile_id": tile_id,
        "target_date": T,
        "variables": variables,
        "dates": dates,
        "context_months": context,
        "gap_months": gap,
        "ignore_label": ignore_label,
        "has_forest_mask": forestmask_catalog is not None,
    }

    return X, y_bin, meta


def balanced_random_spatial_crops(
    X: np.ndarray,
    y: np.ndarray,
    patch_size: int,
    n_patches: int,
    pos_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
    max_negative_tries: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw random spatial crops from a full cube, with control over the
    proportion of patches that contain deforestation (y > 0).

    Parameters
    ----------
    X : np.ndarray
        Input cube of shape [V, T, H, W].
    y : np.ndarray
        Binary target of shape [H, W] (0 = no deforestation, 1 = deforestation).
    patch_size : int
        Spatial size of each patch (e.g. 64).
    n_patches : int
        Total number of patches to sample.
    pos_fraction : float
        Approximate fraction of patches that should contain at least
        one positive pixel (deforestation). E.g. 0.5 → ~50% positive patches.
    rng : np.random.Generator | None
        Optional NumPy RNG for reproducibility.
    max_negative_tries : int
        Maximum number of retries when trying to sample a negative patch
        (patch with no positives).

    Returns
    -------
    X_patches : np.ndarray
        Shape [n_patches, V, T, patch_size, patch_size]
    y_patches : np.ndarray
        Shape [n_patches, patch_size, patch_size]
    """
    if rng is None:
        rng = np.random.default_rng()

    V, T, H, W = X.shape
    ps = patch_size

    if H < ps or W < ps:
        raise ValueError(
            f"Cannot sample {ps}x{ps} patches from cube with spatial size {H}x{W}"
        )

    X_patches = np.empty((n_patches, V, T, ps, ps), dtype=X.dtype)
    y_patches = np.empty((n_patches, ps, ps), dtype=y.dtype)

    # Coordinates of positive pixels (deforestation)
    # y == 1: deforestation, y == 0: no deforestation, y == 2: ignore
    pos_coords = np.argwhere(y == 1)  # shape [N_pos, 2]

    max_row = H - ps
    max_col = W - ps

    def sample_positive_patch() -> tuple[int, int]:
        """
        Sample a patch that contains at least one positive pixel,
        by choosing a positive pixel and placing a patch around it.
        """
        if pos_coords.size == 0:
            # No positives at all in this tile → fall back to negative sampling
            return sample_negative_patch()

        r_pos, c_pos = pos_coords[rng.integers(0, len(pos_coords))]

        # Choose a top-left (r0, c0) so that:
        # - patch stays within [0, H-ps] x [0, W-ps]
        # - (r_pos, c_pos) lies inside the patch
        r_min = max(0, r_pos - ps + 1)
        r_max = min(r_pos, H - ps)
        c_min = max(0, c_pos - ps + 1)
        c_max = min(c_pos, W - ps)

        r0 = rng.integers(r_min, r_max + 1)
        c0 = rng.integers(c_min, c_max + 1)

        return int(r0), int(c0)

    def sample_negative_patch() -> tuple[int, int]:
        """
        Sample a patch with no positive pixels (if possible).
        """
        for _ in range(max_negative_tries):
            r0 = rng.integers(0, max_row + 1)
            c0 = rng.integers(0, max_col + 1)

            patch = y[r0: r0 + ps, c0: c0 + ps]
            # Negative patch: no deforestation pixels (y==1).
            # It's allowed to contain ignore pixels (y==2) – those will be masked in the loss.
            if not np.any(patch == 1):
                return int(r0), int(c0)

        # Fallback: accept whatever we get (may contain positives)
        r0 = rng.integers(0, max_row + 1)
        c0 = rng.integers(0, max_col + 1)
        return int(r0), int(c0)

    for i in range(n_patches):
        # Decide whether we want a "positive" or "negative" patch
        want_pos = rng.random() < pos_fraction

        if want_pos:
            r0, c0 = sample_positive_patch()
        else:
            r0, c0 = sample_negative_patch()

        X_patches[i] = X[:, :, r0 : r0 + ps, c0 : c0 + ps]
        y_patches[i] = y[r0 : r0 + ps, c0 : c0 + ps]

    return X_patches, y_patches


def random_spatial_crops(
    X: np.ndarray,
    y: np.ndarray,
    patch_size: int,
    n_patches: int,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw *unbiased* random spatial crops from a cube, no class balancing.
    This is what we want for val/test so that label frequencies mirror reality.
    """
    if rng is None:
        rng = np.random.default_rng()

    V, T, H, W = X.shape
    ps = patch_size

    if H < ps or W < ps:
        raise ValueError(
            f"Cannot sample {ps}x{ps} patches from cube with spatial size {H}x{W}"
        )

    X_patches = np.empty((n_patches, V, T, ps, ps), dtype=X.dtype)
    y_patches = np.empty((n_patches, ps, ps), dtype=y.dtype)

    max_row = H - ps
    max_col = W - ps

    for i in range(n_patches):
        r0 = rng.integers(0, max_row + 1)
        c0 = rng.integers(0, max_col + 1)

        X_patches[i] = X[:, :, r0 : r0 + ps, c0 : c0 + ps]
        y_patches[i] = y[r0 : r0 + ps, c0 : c0 + ps]

    return X_patches, y_patches

