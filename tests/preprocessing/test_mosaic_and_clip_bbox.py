import numpy as np
import rasterio
from rasterio.transform import from_origin

from deforestation_predictor.preprocessing.mosaic_and_clip_bbox import (
    _mosaic_and_clip_group,
    mosaic_and_clip_region,
)


def _write_tif(path, array, *, transform=None, dtype="float32"):
    """
    Helper to write a single-band GeoTIFF.

    Parameters
    ----------
    path : Path-like
        Output path.
    array : array-like
        2D data array.
    transform : Affine, optional
        Geo-transform. If None, a simple 1x1 grid at origin is used.
    dtype : str
        Raster dtype.
    """
    array = np.asarray(array).astype(dtype)
    h, w = array.shape

    if transform is None:
        transform = from_origin(0, h, 1, 1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(array, 1)


def test_mosaic_and_clip_group_two_tiles_full_bounds(tmp_path):
    """
    _mosaic_and_clip_group should correctly mosaic two adjacent tiles and
    clip to the full combined extent.
    """
    # Tile A: x in [0, 2), y in [0, 2)
    a_path = tmp_path / "tileA.tif"
    a_arr = np.ones((2, 2), dtype=np.float32)
    a_transform = from_origin(0, 2, 1, 1)
    _write_tif(a_path, a_arr, transform=a_transform)

    # Tile B: x in [2, 4), y in [0, 2) — directly east of A
    b_path = tmp_path / "tileB.tif"
    b_arr = np.full((2, 2), 2, dtype=np.float32)
    b_transform = from_origin(2, 2, 1, 1)
    _write_tif(b_path, b_arr, transform=b_transform)

    out_path = tmp_path / "mosaic_full.tif"

    # Full combined bounds of both tiles
    bounds = (0.0, 0.0, 4.0, 2.0)  # (west, south, east, north)

    _mosaic_and_clip_group([a_path, b_path], bounds, out_path)

    with rasterio.open(out_path) as src:
        data = src.read(1)
        assert data.shape == (2, 4)

        # Left half from tile A (1s), right half from tile B (2s)
        assert np.all(data[:, :2] == 1)
        assert np.all(data[:, 2:] == 2)


def test_mosaic_and_clip_group_clipped_subwindow(tmp_path):
    """
    _mosaic_and_clip_group should respect a smaller bounding box and
    return only the requested sub-window.
    """
    # Same setup as previous test
    a_path = tmp_path / "tileA.tif"
    a_arr = np.ones((2, 2), dtype=np.float32)
    a_transform = from_origin(0, 2, 1, 1)
    _write_tif(a_path, a_arr, transform=a_transform)

    b_path = tmp_path / "tileB.tif"
    b_arr = np.full((2, 2), 2, dtype=np.float32)
    b_transform = from_origin(2, 2, 1, 1)
    _write_tif(b_path, b_arr, transform=b_transform)

    out_path = tmp_path / "mosaic_clipped.tif"

    # Bounds that cut off one column on each side:
    # x in [1, 3) => one column from A, one from B
    bounds = (1.0, 0.0, 3.0, 2.0)

    _mosaic_and_clip_group([a_path, b_path], bounds, out_path)

    with rasterio.open(out_path) as src:
        data = src.read(1)

        # 2 rows, 2 columns (one from each original tile)
        assert data.shape == (2, 2)

        # First column from tile A (1s), second from tile B (2s)
        assert np.all(data[:, 0] == 1)
        assert np.all(data[:, 1] == 2)


def test_mosaic_and_clip_region_builds_gabon_like_region(tmp_path):
    """
    mosaic_and_clip_region should:
      - read multi-tile input and GT tiles
      - mosaic them per (date, variable)
      - clip to the given bounds
      - write new files with tile_id = region_id
    """
    # Simulate your folder structure under a temporary root
    raw_input_root = tmp_path / "input"
    raw_gt_root = tmp_path / "groundtruth"

    # Two tiles that will be mosaiced into a single region
    tile_ids = ["00N_000E", "10N_000E"]
    date_str = "2023-01-01"

    for i, tile_id in enumerate(tile_ids):
        # Input tile directory
        in_tile_dir = raw_input_root / tile_id
        in_tile_dir.mkdir(parents=True, exist_ok=True)

        # GT tile directory
        gt_tile_dir = raw_gt_root / tile_id
        gt_tile_dir.mkdir(parents=True, exist_ok=True)

        # Offset so tiles are adjacent in x-direction
        x_offset = i * 2.0  # 0 for first tile, 2 for second
        transform = from_origin(x_offset, 2.0, 1.0, 1.0)

        # Input variable (non-GT)
        in_fname = f"{tile_id}_{date_str}_varA.tif"
        in_path = in_tile_dir / in_fname
        in_value = float(i + 1)  # 1 for first tile, 2 for second
        _write_tif(
            in_path,
            np.full((2, 2), in_value, dtype=np.float32),
            transform=transform,
        )

        # GT variable, must be one of {"gt", "groundtruth6m", "groundtruth"}
        gt_fname = f"{tile_id}_{date_str}_gt.tif"
        gt_path = gt_tile_dir / gt_fname
        _write_tif(
            gt_path,
            np.full((2, 2), in_value, dtype=np.float32),
            transform=transform,
        )

    out_input_root = tmp_path / "processed_input"
    out_gt_root = tmp_path / "processed_gt"

    # Bounds that cover both tiles (x in [0,4), y in [0,2))
    bounds = (0.0, 0.0, 4.0, 2.0)

    region_id = "GABON"

    mosaic_and_clip_region(
        raw_input_root=raw_input_root,
        raw_gt_root=raw_gt_root,
        out_input_root=out_input_root,
        out_gt_root=out_gt_root,
        region_id=region_id,
        bounds=bounds,
        tile_ids=tile_ids,
    )

    # Check that the mosaiced region files exist
    region_input_dir = out_input_root / region_id
    region_gt_dir = out_gt_root / region_id

    input_files = list(region_input_dir.glob("*.tif"))
    gt_files = list(region_gt_dir.glob("*.tif"))

    # One mosaiced file for input and one for GT
    assert len(input_files) == 1
    assert len(gt_files) == 1

    # Filenames should start with region_id and the same date
    assert input_files[0].name.startswith(f"{region_id}_{date_str}_")
    assert gt_files[0].name.startswith(f"{region_id}_{date_str}_")

    # Check mosaiced input content: left half from first tile, right from second
    with rasterio.open(input_files[0]) as src:
        data = src.read(1)
        assert data.shape == (2, 4)
        assert np.all(data[:, :2] == 1)  # from 00N_000E
        assert np.all(data[:, 2:] == 2)  # from 10N_000E
