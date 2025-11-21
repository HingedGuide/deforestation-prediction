from deforestation_predictor.preprocessing.builder import (
    stack_rasters,
    normalize_cube_auto,
    build_sample,
    balanced_random_spatial_crops,
)

from deforestation_predictor.preprocessing.catalog import (
    build_gt_catalog,
    build_raster_catalog,
    get_records_for_dates,
    compute_variable_maxima,
)

import pytest
import numpy as np
from datetime import datetime
import rasterio
import pandas as pd
from rasterio.transform import from_origin


def _write_tif(path, array, dtype="float32"):
    """Test helper: write a single-band GeoTIFF with a dummy geotransform/CRS."""
    array = np.asarray(array).astype(dtype)
    h, w = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(array, 1)

def test_stack_rasters(tmp_path):
    # Create mock .tif files with known data
    data_shape = (5, 5)  # H, W

    dates = [
        datetime(2023, 8, 1),
        datetime(2023, 9, 1),
        datetime(2023, 10, 1),
    ]
    variables = ["wetlands", "elevation"]

    # Give each (date, variable) a unique constant value
    value_map = {}

    for t_idx, d in enumerate(dates):
        for v_idx, v in enumerate(variables):
            value = 10 * (t_idx + 1) + (v_idx + 1)  # e.g. 11, 12, 21, 22, ...
            value_map[(d, v)] = value

            fname = f"00N_010E_{d.date()}_{v}.tif"
            data = np.full(data_shape, value, dtype=np.uint8)

            with rasterio.open(
                tmp_path / fname,
                "w",
                driver="GTiff",
                height=data_shape[0],
                width=data_shape[1],
                count=1,
                dtype=data.dtype,
            ) as dst:
                dst.write(data, 1)

    catalog = build_raster_catalog(str(tmp_path))
    subset = get_records_for_dates(
        catalog,
        "00N_010E",
        datetime(2023, 8, 1),
        datetime(2023, 10, 1),
    )

    cube, vars_out, dates_out = stack_rasters(subset)

    # V = 2 (wetlands, elevation), T = 3, H = 5, W = 5
    assert cube.shape == (2, 3, 5, 5)

    # Dates should be in chronological order
    assert dates_out == dates

    # We don't assume a specific variable axis order, we look it up
    assert set(vars_out) == set(variables)
    wet_idx = vars_out.index("wetlands")
    elev_idx = vars_out.index("elevation")

    # wetlands on 2023-08-01 (time index 0)
    expected = value_map[(datetime(2023, 8, 1), "wetlands")]
    assert np.array_equal(
        cube[wet_idx, 0], np.full(data_shape, expected, dtype=cube.dtype)
    )

    # wetlands on 2023-09-01 (time index 1)
    expected = value_map[(datetime(2023, 9, 1), "wetlands")]
    assert np.array_equal(
        cube[wet_idx, 1], np.full(data_shape, expected, dtype=cube.dtype)
    )

    # elevation on 2023-10-01 (time index 2)
    expected = value_map[(datetime(2023, 10, 1), "elevation")]
    assert np.array_equal(
        cube[elev_idx, 2], np.full(data_shape, expected, dtype=cube.dtype)
    )


def test_normalize_cube_auto(tmp_path):
    """
    Check: division by auto maxima, NaN/Inf -> 0, clipping to [0,1],
    and optional uint8 casting path.
    """
    # Create tiny dataset for two variables
    a1 = tmp_path / "00N_010E_2023-01-01_varA.tif"
    _write_tif(a1, [[1.0, 2.0], [np.nan, 10.0]])

    a2 = tmp_path / "00N_010E_2023-02-01_varA.tif"
    _write_tif(a2, [[3.0, 4.0], [5.0, np.inf]])

    b1 = tmp_path / "00N_010E_2023-01-01_varB.tif"
    _write_tif(b1, [[-1.0, 0.0], [6.0, 7.0]])

    b2 = tmp_path / "00N_010E_2023-02-01_varB.tif"
    _write_tif(b2, [[-2.0, -3.0], [np.nan, 5.0]])

    catalog = build_raster_catalog(str(tmp_path))

    # Build a cube [V=2, T=2, H=2, W=2] in variable order ["varA", "varB"]
    variables = ["varA", "varB"]
    X = np.array(
        [
            # varA (two time steps)
            [
                [[1.0, 2.0], [np.nan, 10.0]],
                [[3.0, 4.0], [5.0, np.inf]],
            ],
            # varB (two time steps)
            [
                [[-1.0, 0.0], [6.0, 7.0]],
                [[-2.0, -3.0], [np.nan, 5.0]],
            ],
        ],
        dtype=np.float32,
    )

    # 1) Float output, overflow clipped
    Xn = normalize_cube_auto(
        X,
        variables=variables,
        catalog=catalog,
        nan_policy="zero",
        overflow="clip",
        cast=None,
    )

    # Expected varA: divide by 10, NaN/Inf -> 0, clip
    exp_a_t1 = np.array([[0.1, 0.2], [0.0, 1.0]], dtype=np.float32)
    exp_a_t2 = np.array([[0.3, 0.4], [0.5, 0.0]], dtype=np.float32)

    # Expected varB: divide by 7, negatives -> clipped to 0, NaN->0
    exp_b_t1 = np.array([[0.0, 0.0], [6/7, 1.0]], dtype=np.float32)
    exp_b_t2 = np.array([[0.0, 0.0], [0.0, 5/7]], dtype=np.float32)

    assert np.allclose(Xn[0, 0], exp_a_t1, atol=1e-6)
    assert np.allclose(Xn[0, 1], exp_a_t2, atol=1e-6)
    assert np.allclose(Xn[1, 0], exp_b_t1, atol=1e-6)
    assert np.allclose(Xn[1, 1], exp_b_t2, atol=1e-6)

    # 2) Same normalization but cast to uint8 (ensures [0,1]→[0,255] path works)
    Xn_u8 = normalize_cube_auto(
        X,
        variables=variables,
        catalog=catalog,
        nan_policy="zero",
        overflow="clip",
        cast="uint8",
    )
    exp_u8 = np.stack(
        [
            np.stack(
                [
                    (exp_a_t1 * 255).round().astype(np.uint8),
                    (exp_a_t2 * 255).round().astype(np.uint8),
                ],
                axis=0,
            ),
            np.stack(
                [
                    (exp_b_t1 * 255).round().astype(np.uint8),
                    (exp_b_t2 * 255).round().astype(np.uint8),
                ],
                axis=0,
            ),
        ],
        axis=0,
    )
    assert Xn_u8.dtype == np.uint8
    assert np.array_equal(Xn_u8, exp_u8)


def test_build_sample(tmp_path):
    data_shape = (5, 5)  # H, W

    # Input dates for a small window: context=2, gap=1, T=2023-10-01
    input_dates = [
        datetime(2023, 8, 1),
        datetime(2023, 9, 1),
    ]
    dyn_vars = ["wetlands", "elevation"]
    static_var = "slope"

    # ---- Create dynamic input rasters for both dates and variables ----
    for d in input_dates:
        for v in dyn_vars:
            fname = f"00N_010E_{d.date()}_{v}.tif"
            data = np.random.randint(0, 255, size=data_shape, dtype=np.uint8)

            with rasterio.open(
                tmp_path / fname,
                "w",
                driver="GTiff",
                height=data_shape[0],
                width=data_shape[1],
                count=1,
                dtype=data.dtype,
            ) as dst:
                dst.write(data, 1)

    # ---- Create a static raster (e.g. slope) with a known constant value ----
    slope_value = 5.0
    slope_fname = tmp_path / "00N_010E_2021-01-01_slope.tif"
    _write_tif(slope_fname, np.full(data_shape, slope_value, dtype=np.float32))

    # ---- Create GT raster for target_date ----
    target_date = datetime(2023, 10, 1)
    gt_fname = f"00N_010E_{target_date.date()}_gt.tif"
    gt_data = np.random.randint(0, 2, size=data_shape, dtype=np.uint8)

    with rasterio.open(
        tmp_path / gt_fname,
        "w",
        driver="GTiff",
        height=data_shape[0],
        width=data_shape[1],
        count=1,
        dtype=gt_data.dtype,
    ) as dst:
        dst.write(gt_data, 1)

    # ---- Build catalogs ----
    full_catalog = build_raster_catalog(str(tmp_path))
    gt_catalog = build_gt_catalog(str(tmp_path))

    # Dynamic catalog: only the monthly vars
    dyn_catalog = full_catalog[full_catalog["variable"].isin(dyn_vars)].reset_index(drop=True)

    # Static catalog: only slope
    static_catalog = full_catalog[full_catalog["variable"] == static_var].reset_index(drop=True)

    # Maxima for normalization (both dynamic + static)
    maxima = compute_variable_maxima(full_catalog)

    # ---- Build sample with static_catalog + maxima ----
    X, y, meta = build_sample(
        "00N_010E",
        target_date,
        dyn_catalog,
        gt_catalog,
        context=2,
        gap=1,
        static_catalog=static_catalog,
        maxima=maxima,
    )

    # V = dynamic vars + static vars = 2 + 1 = 3, T = 2
    expected_V = len(dyn_vars) + 1
    assert X.shape == (expected_V, 2, data_shape[0], data_shape[1])

    # GT shape
    assert y.shape == data_shape

    # Meta sanity checks
    assert meta["tile_id"] == "00N_010E"
    assert meta["target_date"] == pd.to_datetime(target_date)

    expected_vars = set(dyn_vars + [static_var])
    assert set(meta["variables"]) == expected_vars

    assert len(meta["dates"]) == 2
    assert meta["dates"][0] == datetime(2023, 8, 1)
    assert meta["dates"][1] == datetime(2023, 9, 1)

    # ---- Check that the static channel is broadcast over time and normalized ----
    slope_idx = meta["variables"].index(static_var)

    # maxima['slope'] should be slope_value (only one raster with value 5.0)
    assert maxima[static_var] == pytest.approx(slope_value)

    # After normalization: slope array should be all ones, same for both timesteps
    expected_static = np.ones(data_shape, dtype=np.float32)

    assert np.allclose(X[slope_idx, 0], expected_static)
    assert np.allclose(X[slope_idx, 1], expected_static)


def test_build_sample_applies_forest_mask_ignore_label(tmp_path):
    """
    Check that build_sample uses the initialforestcover mask to set
    an ignore_label (2) outside forest, and leaves forest pixels as
    0/1 according to the GT.
    """
    data_shape = (4, 4)
    tile_id = "00N_010E"

    # Input dates for a small window: context=2, gap=1, T=2023-10-01
    input_dates = [
        datetime(2023, 8, 1),
        datetime(2023, 9, 1),
    ]
    dyn_var = "wetlands"

    # ---- Dynamic input rasters ----
    for d in input_dates:
        fname = f"{tile_id}_{d.date()}_{dyn_var}.tif"
        data = np.random.randint(0, 255, size=data_shape, dtype=np.uint8)
        with rasterio.open(
            tmp_path / fname,
            "w",
            driver="GTiff",
            height=data_shape[0],
            width=data_shape[1],
            count=1,
            dtype=data.dtype,
        ) as dst:
            dst.write(data, 1)

    # ---- GT raster for target_date ----
    target_date = datetime(2023, 10, 1)
    gt_fname = f"{tile_id}_{target_date.date()}_gt.tif"
    # Binary GT (0/1) so binarization in build_sample is predictable
    gt_data = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    with rasterio.open(
        tmp_path / gt_fname,
        "w",
        driver="GTiff",
        height=data_shape[0],
        width=data_shape[1],
        count=1,
        dtype=gt_data.dtype,
    ) as dst:
        dst.write(gt_data, 1)

    # ---- Forest mask from initialforestcover ----
    # Values > 2000 are treated as "forest" in the current implementation
    forest_raw = np.array(
        [
            [2500, 1500,    0, 5000],
            [2100, 1999, 2001,    0],
            [   0,    0, 3000, 4000],
            [1000, 1000, 9999, 2000],
        ],
        dtype=np.float32,
    )
    mask_fname = tmp_path / f"{tile_id}_2020-01-01_initialforestcover.tif"
    _write_tif(mask_fname, forest_raw, dtype="float32")

    # ---- Build catalogs ----
    full_catalog = build_raster_catalog(str(tmp_path))
    gt_catalog = build_gt_catalog(str(tmp_path))

    dyn_catalog = full_catalog[full_catalog["variable"] == dyn_var].reset_index(drop=True)
    forestmask_catalog = full_catalog[
        full_catalog["variable"] == "initialforestcover"
    ].reset_index(drop=True)

    # ---- Call build_sample with forestmask_catalog ----
    X, y, meta = build_sample(
        tile_id,
        target_date,
        dyn_catalog,
        gt_catalog,
        context=2,
        gap=1,
        static_catalog=None,
        normalize=False,              # we don't care about normalization here
        maxima=None,
        forestmask_catalog=forestmask_catalog,
        mask_threshold=2000.0,
    )

    # Shape sanity checks
    assert X.shape[2:] == data_shape  # H, W
    assert y.shape == data_shape

    # Expected behaviour: y == 1 where GT==1 *and* forest_raw > 2000,
    # y == 0 where GT==0 *and* forest_raw > 2000,
    # y == 2 (ignore_label) where forest_raw <= 2000
    ignore_label = 2
    forest_mask = forest_raw > 2000.0
    expected_y = np.where(forest_mask, gt_data.astype(np.uint8), ignore_label)

    assert np.array_equal(y, expected_y)

    # Meta fields reflect mask usage
    assert meta["ignore_label"] == ignore_label
    assert meta["has_forest_mask"] is True


def test_balanced_random_spatial_crops_shapes_and_balance():
    """
    Basic sanity check: shapes are correct and the fraction of
    positive patches roughly matches pos_fraction.
    """
    rng = np.random.default_rng(0)

    V, T, H, W = 2, 3, 64, 64
    X = rng.normal(size=(V, T, H, W)).astype(np.float32)

    # Make a small positive region somewhere in the middle
    y = np.zeros((H, W), dtype=np.uint8)
    y[20:30, 20:30] = 1

    patch_size = 16
    n_patches = 50
    pos_fraction = 0.6

    X_patches, y_patches = balanced_random_spatial_crops(
        X,
        y,
        patch_size=patch_size,
        n_patches=n_patches,
        pos_fraction=pos_fraction,
        rng=rng,
    )

    # Shape checks
    assert X_patches.shape == (n_patches, V, T, patch_size, patch_size)
    assert y_patches.shape == (n_patches, patch_size, patch_size)

    # Fraction of patches that contain at least one positive pixel
    has_pos = (y_patches > 0).any(axis=(1, 2))
    pos_ratio = has_pos.mean()

    # With n_patches=50 this should be close-ish to 0.6
    # Allow some slack because of randomness
    assert 0.4 <= pos_ratio <= 0.8, f"pos_ratio={pos_ratio} outside expected range"


def test_balanced_random_spatial_crops_no_positives_all_negative():
    """
    When there are no positive pixels in y, all patches should be negative,
    regardless of pos_fraction.
    """
    rng = np.random.default_rng(1)

    V, T, H, W = 2, 2, 32, 32
    X = rng.normal(size=(V, T, H, W)).astype(np.float32)
    y = np.zeros((H, W), dtype=np.uint8)  # no positives at all

    patch_size = 8
    n_patches = 20

    X_patches, y_patches = balanced_random_spatial_crops(
        X,
        y,
        patch_size=patch_size,
        n_patches=n_patches,
        pos_fraction=0.9,  # ask for mostly positives, but there are none
        rng=rng,
    )

    # Shapes still fine
    assert X_patches.shape == (n_patches, V, T, patch_size, patch_size)
    assert y_patches.shape == (n_patches, patch_size, patch_size)

    # All patches should be zero
    assert not (y_patches > 0).any(), "Found positive pixels in y_patches but y had no positives"


def test_balanced_random_spatial_crops_too_large_patch_raises():
    """
    If the requested patch_size is larger than the spatial dimension,
    the function should raise a ValueError.
    """
    rng = np.random.default_rng(2)

    V, T, H, W = 1, 1, 16, 16
    X = rng.normal(size=(V, T, H, W)).astype(np.float32)
    y = np.zeros((H, W), dtype=np.uint8)

    with pytest.raises(ValueError):
        balanced_random_spatial_crops(
            X,
            y,
            patch_size=32,   # larger than H/W
            n_patches=5,
            rng=rng,
        )
