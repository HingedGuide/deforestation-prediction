from preprocess_images import *


def test_parse_filename():
    filename = "00N_010E_2023-08-15_wetlands.tif"
    parsed = parse_filename(filename)

    expected = {
        "tile_id": "00N_010E",
        "date": datetime(2023, 8, 15),
        "variable": "wetlands",
        "path": filename
    }

    assert parsed == expected


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


def test_get_input_window_range():
    target_date = datetime(2023, 10, 1)

    # Small example: context=3, gap=1 → [T-3, T-1]
    start, end = get_input_window_range(target_date, context=3, gap=1)
    assert start == datetime(2023, 7, 1)
    assert end == datetime(2023, 9, 1)

    # Default example: context=12, gap=3
    start, end = get_input_window_range(target_date)
    # start = T - (12+3-1) = T - 14 months
    assert start == datetime(2022, 8, 1)
    # end = T - 3 months
    assert end == datetime(2023, 7, 1)


def test_build_sample(tmp_path):
    data_shape = (5, 5)  # H, W

    # Input dates for a small window: context=2, gap=1, T=2023-10-01
    input_dates = [
        datetime(2023, 8, 1),
        datetime(2023, 9, 1),
    ]
    variables = ["wetlands", "elevation"]

    # Create input rasters for both dates and variables
    for d in input_dates:
        for v in variables:
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

    # Create GT raster for target_date
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

    catalog = build_raster_catalog(str(tmp_path))
    gt_catalog = build_gt_catalog(str(tmp_path))

    X, y, meta = build_sample(
        "00N_010E",
        target_date,
        catalog,
        gt_catalog,
        context=2,
        gap=1,
    )

    # Input cube shape [V, T, H, W]
    assert X.shape == (len(variables), 2, data_shape[0], data_shape[1])

    # GT shape
    assert y.shape == data_shape

    # Meta sanity checks
    assert meta["tile_id"] == "00N_010E"
    assert meta["target_date"] == pd.to_datetime(target_date)
    assert set(meta["variables"]) == set(variables)
    assert len(meta["dates"]) == 2
    assert meta["dates"][0] == datetime(2023, 8, 1)
    assert meta["dates"][1] == datetime(2023, 9, 1)
