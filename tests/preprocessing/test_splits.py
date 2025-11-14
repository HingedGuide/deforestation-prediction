from datetime import datetime

import pandas as pd

from deforestation_predictor.preprocessing.splits import (
    build_target_table,
    has_full_window,
    filter_targets_with_full_window,
    split_targets_by_time,
    TemporalSplitConfig,
)
from deforestation_predictor.preprocessing.windows import get_input_window_range


def test_build_target_table_basic():
    # Create a GT catalog with duplicates and out-of-range dates
    data = {
        "tile_id": ["A", "A", "A", "B"],
        "date": [
            datetime(2023, 1, 1),
            datetime(2023, 1, 1),  # duplicate
            datetime(2023, 6, 1),
            datetime(2022, 12, 1),
        ],
        "variable": ["gt", "gt", "gt", "gt"],
        "path": ["p1", "p1", "p2", "p3"],
    }
    gt_catalog = pd.DataFrame(data)

    targets = build_target_table(
        gt_catalog,
        min_date="2023-01-01",
        max_date="2023-12-31",
    )

    # We expect unique (tile_id, date) pairs in range [2023-01-01, 2023-12-31]
    assert list(targets.columns) == ["tile_id", "date"]
    assert len(targets) == 2

    # Should contain:
    # A, 2023-01-01
    # A, 2023-06-01
    assert (targets["tile_id"] == "A").all()
    assert set(targets["date"]) == {
        datetime(2023, 1, 1),
        datetime(2023, 6, 1),
    }


def test_has_full_window_true_and_false():
    # Build a small synthetic input catalog:
    # tile "T", variables "varA", "varB", monthly from 2023-01 to 2023-04
    rows = []
    tile_id = "T"
    variables = ["varA", "varB"]
    dates = pd.date_range("2023-01-01", "2023-04-01", freq="MS")

    for d in dates:
        for v in variables:
            rows.append(
                {
                    "tile_id": tile_id,
                    "date": d,
                    "variable": v,
                    "path": f"/fake/{tile_id}_{d.date()}_{v}.tif",
                }
            )

    catalog = pd.DataFrame(rows)

    # Choose context=2, gap=1, target_date=2023-04-01
    # -> window is from 2023-02-01 to 2023-03-01
    context = 2
    gap = 1
    target_date = datetime(2023, 4, 1)

    start, end = get_input_window_range(target_date, context=context, gap=gap)
    assert start == pd.Timestamp("2023-02-01")
    assert end == pd.Timestamp("2023-03-01")

    # Full window should be available
    assert has_full_window(tile_id, target_date, catalog, context=context, gap=gap)

    # Now remove one record from the catalog to break the window
    # e.g. missing varB at 2023-02-01
    broken_catalog = catalog[
        ~(
            (catalog["date"] == pd.Timestamp("2023-02-01"))
            & (catalog["variable"] == "varB")
        )
    ].copy()

    assert not has_full_window(
        tile_id,
        target_date,
        broken_catalog,
        context=context,
        gap=gap,
    )


def test_filter_targets_with_full_window():
    # Catalog with 3 possible target dates
    tile_id = "T"
    variables = ["varA", "varB"]

    # Input range: 2023-01 to 2023-05
    input_dates = pd.date_range("2023-01-01", "2023-05-01", freq="MS")
    rows = []
    for d in input_dates:
        for v in variables:
            rows.append(
                {
                    "tile_id": tile_id,
                    "date": d,
                    "variable": v,
                    "path": f"/fake/{tile_id}_{d.date()}_{v}.tif",
                }
            )
    catalog = pd.DataFrame(rows)

    # Candidate targets at 2023-04-01, 2023-05-01, 2023-06-01
    targets = pd.DataFrame(
        {
            "tile_id": [tile_id, tile_id, tile_id],
            "date": [
                datetime(2023, 4, 1),
                datetime(2023, 5, 1),
                datetime(2023, 6, 1),
            ],
        }
    )

    context = 2
    gap = 1

    # For context=2, gap=1:
    # T=2023-04-01 -> window 2023-02-01..2023-03-01 -> fully covered
    # T=2023-05-01 -> window 2023-03-01..2023-04-01 -> fully covered
    # T=2023-06-01 -> window 2023-04-01..2023-05-01 -> fully covered
    filtered = filter_targets_with_full_window(
        targets,
        catalog,
        context=context,
        gap=gap,
    )
    assert len(filtered) == 3

    # Now break the catalog: remove the month 2023-02 so that
    # the window for T=2023-04-01 (which needs 2023-02 and 2023-03)
    # becomes incomplete.
    broken_catalog = catalog[catalog["date"] != pd.Timestamp("2023-02-01")].copy()

    filtered_broken = filter_targets_with_full_window(
        targets,
        broken_catalog,
        context=context,
        gap=gap,
    )

    # Now T=2023-04-01 is invalid (missing 2023-02-01 inputs),
    # but T=2023-05-01 and T=2023-06-01 still have their windows.
    assert len(filtered_broken) == 2
    assert set(filtered_broken["date"]) == {
        datetime(2023, 5, 1),
        datetime(2023, 6, 1),
    }


def test_split_targets_by_time():
    # Build a simple target table with mixed dates
    data = {
        "tile_id": ["A", "A", "B", "B", "C"],
        "date": [
            datetime(2022, 12, 1),
            datetime(2023, 3, 1),
            datetime(2023, 9, 1),
            datetime(2024, 1, 1),
            datetime(2024, 6, 1),
        ],
    }
    targets = pd.DataFrame(data).sort_values("date").reset_index(drop=True)

    cfg = TemporalSplitConfig(
        train_end="2022-12-31",
        val_end="2023-12-31",
        context=12,
        gap=3,
    )

    train_targets, val_targets, test_targets = split_targets_by_time(targets, cfg)

    # train: date <= 2022-12-31
    assert len(train_targets) == 1
    assert train_targets["date"].iloc[0] == datetime(2022, 12, 1)

    # val: 2022-12-31 < date <= 2023-12-31
    assert len(val_targets) == 2
    assert set(val_targets["date"]) == {
        datetime(2023, 3, 1),
        datetime(2023, 9, 1),
    }

    # test: date > 2023-12-31
    assert len(test_targets) == 2
    assert set(test_targets["date"]) == {
        datetime(2024, 1, 1),
        datetime(2024, 6, 1),
    }
