from deforestation_predictor.preprocessing.catalog import build_raster_catalog, build_gt_catalog, get_records_for_dates, compute_variable_maxima

from datetime import datetime
from pathlib import Path
import numpy as np
import rasterio
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

def test_build_raster_catalog(tmp_path):
    # Create mock .tif files
    filenames = [
        "00N_010E_2023-08-15_wetlands.tif",
        "00N_010E_2023-08-16_wetlands.tif",
        "00N_010E_2023-08-15_elevation.tif",
        "01N_011E_2023-08-15_wetlands.tif"
    ]

    for fname in filenames:
        (tmp_path / fname).touch()

    df = build_raster_catalog(str(tmp_path))

    assert len(df) == 4
    assert set(df['tile_id']) == {"00N_010E", "01N_011E"}
    assert set(df['variable']) == {"wetlands", "elevation"}
    assert all(isinstance(d, datetime) for d in df['date'])


def test_build_gt_catalog(tmp_path):
    # Create mock .tif files
    filenames = [
        "00N_010E_2023-08-15_gt.tif",
        "00N_010E_2023-09-15_gt.tif",
        "01N_011E_2023-08-15_gt.tif",
    ]

    for fname in filenames:
        (tmp_path / fname).touch()

    df = build_gt_catalog(str(tmp_path))

    assert len(df) == 3
    assert set(df["tile_id"]) == {"00N_010E", "01N_011E"}
    assert all(isinstance(d, datetime) for d in df["date"])
    assert set(df["variable"]) == {"gt"}


def test_get_records_for_dates(tmp_path):
    # Create mock .tif files
    filenames = [
        "00N_010E_2023-08-01_wetlands.tif",
        "00N_010E_2023-09-01_wetlands.tif",
        "00N_010E_2023-10-01_elevation.tif",
        "01N_011E_2023-08-15_wetlands.tif"
    ]

    for fname in filenames:
        (tmp_path / fname).touch()

    catalog = build_raster_catalog(str(tmp_path))

    start = datetime(2023, 8, 1)
    end = datetime(2023, 10, 1)

    subset = get_records_for_dates(catalog, "00N_010E", start, end)

    assert len(subset) == 3
    assert all(subset['tile_id'] == "00N_010E")


def test_compute_variable_maxima(tmp_path):
    """
    varA across 2 files: max should be 10 (NaN/Inf ignored)
    varB across 2 files: max should be 7
    """
    # varA
    a1 = tmp_path / "00N_010E_2023-01-01_varA.tif"
    _write_tif(a1, [[1, 2], [np.nan, 10]])
    a2 = tmp_path / "00N_010E_2023-02-01_varA.tif"
    _write_tif(a2, [[3, 4], [5, np.inf]])

    # varB
    b1 = tmp_path / "00N_010E_2023-01-01_varB.tif"
    _write_tif(b1, [[-1, 0], [6, 7]])
    b2 = tmp_path / "00N_010E_2023-02-01_varB.tif"
    _write_tif(b2, [[-2, -3], [np.nan, 5]])

    # Build a minimal catalog DataFrame (reuse your builder if you like)
    catalog = build_raster_catalog(str(tmp_path))

    maxima = compute_variable_maxima(catalog)
    assert np.isclose(maxima["varA"], 10.0)
    assert np.isclose(maxima["varB"], 7.0)