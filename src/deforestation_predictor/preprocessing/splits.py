from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from deforestation_predictor.preprocessing.catalog import get_records_for_dates
from deforestation_predictor.preprocessing.windows import (
    CONTEXT_MONTHS,
    GAP_MONTHS,
    get_input_window_range,
)


def build_target_table(
    gt_catalog: pd.DataFrame,
    min_date: str | None = None,
    max_date: str | None = None,
) -> pd.DataFrame:
    """
    Build a table of unique (tile_id, date) pairs from the GT catalog.

    Optionally filter by [min_date, max_date] (inclusive).

    Parameters
    ----------
    gt_catalog : DataFrame
        Must have columns: ['tile_id', 'date'] at least.
    min_date : str | None
        Lower bound for dates (e.g. '2021-01-01'). If None, no lower bound.
    max_date : str | None
        Upper bound for dates (e.g. '2024-12-31'). If None, no upper bound.

    Returns
    -------
    DataFrame with columns ['tile_id', 'date'], sorted by tile_id, date.
    """
    targets = gt_catalog[["tile_id", "date"]].drop_duplicates().copy()

    if min_date is not None:
        targets = targets[targets["date"] >= pd.to_datetime(min_date)]
    if max_date is not None:
        targets = targets[targets["date"] <= pd.to_datetime(max_date)]

    targets.sort_values(["tile_id", "date"], inplace=True)
    targets.reset_index(drop=True, inplace=True)
    return targets


def has_full_window(
    tile_id: str,
    target_date: pd.Timestamp,
    catalog: pd.DataFrame,
    context: int = CONTEXT_MONTHS,
    gap: int = GAP_MONTHS,
) -> bool:
    """
    Check if a tile/target_date has a complete temporal input window.

    A window is considered full if:
      - get_records_for_dates(...) returns any rows
      - For every month in the [start, end] window
      - For every variable present in that subset
      - There is at least one record (tile_id, date, variable).

    Parameters
    ----------
    tile_id : str
        Tile identifier (e.g. '00N_010E').
    target_date : Timestamp
        The ground-truth date.
    catalog : DataFrame
        Input catalog with columns ['tile_id', 'date', 'variable', 'path'].
    context : int
        Number of months of context (e.g. 12).
    gap : int
        Gap in months between the last input month and target_date (e.g. 3).

    Returns
    -------
    bool
        True if full window exists, False otherwise.
    """
    start, end = get_input_window_range(target_date, context=context, gap=gap)
    subset = get_records_for_dates(catalog, tile_id, start, end)

    if subset.empty:
        return False

    variables = set(subset["variable"].unique())
    if not variables:
        return False

    # Expected monthly timestamps in the window
    expected_dates = pd.date_range(
        start=start.normalize(),
        end=end.normalize(),
        freq="MS",  # month start
    )

    # If the expected number of steps doesn't match the context, treat as incomplete
    if len(expected_dates) != context:
        return False

    for d in expected_dates:
        for v in variables:
            if subset[(subset["date"] == d) & (subset["variable"] == v)].empty:
                return False

    return True


def filter_targets_with_full_window(
    targets: pd.DataFrame,
    catalog: pd.DataFrame,
    context: int = CONTEXT_MONTHS,
    gap: int = GAP_MONTHS,
) -> pd.DataFrame:
    """
    Filter a target table to only those (tile_id, date) pairs that
    have a complete temporal window in the input catalog.

    Parameters
    ----------
    targets : DataFrame
        Must have columns ['tile_id', 'date'].
    catalog : DataFrame
        Input catalog used to check availability.
    context : int
        Number of context months.
    gap : int
        Gap in months between last input month and target.

    Returns
    -------
    DataFrame
        Filtered subset of targets with full windows.
    """
    mask = []
    for row in targets.itertuples(index=False):
        ok = has_full_window(
            tile_id=row.tile_id,
            target_date=row.date,
            catalog=catalog,
            context=context,
            gap=gap,
        )
        mask.append(ok)

    filtered = targets[mask].reset_index(drop=True)
    return filtered


@dataclass
class TemporalSplitConfig:
    """
    Configuration for time-based train/val/test splits.

    Dates are inclusive:
      - train: date <= train_end
      - val:   train_end < date <= val_end
      - test:  date > val_end
    """
    train_end: str    # e.g. "2022-12-31"
    val_end: str      # e.g. "2023-12-31"
    context: int = CONTEXT_MONTHS
    gap: int = GAP_MONTHS


def split_targets_by_time(
    targets: pd.DataFrame,
    cfg: TemporalSplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a target table into train/val/test subsets based on date.

    Parameters
    ----------
    targets : DataFrame
        Must have columns ['tile_id', 'date'].
    cfg : TemporalSplitConfig
        Contains train_end, val_end, and context/gap (context/gap unused here,
        but useful for consistent configuration).

    Returns
    -------
    (train_targets, val_targets, test_targets) : tuple of DataFrames
    """
    train_end = pd.to_datetime(cfg.train_end)
    val_end = pd.to_datetime(cfg.val_end)

    train_mask = targets["date"] <= train_end
    val_mask = (targets["date"] > train_end) & (targets["date"] <= val_end)
    test_mask = targets["date"] > val_end

    train_targets = targets[train_mask].reset_index(drop=True)
    val_targets = targets[val_mask].reset_index(drop=True)
    test_targets = targets[test_mask].reset_index(drop=True)

    return train_targets, val_targets, test_targets
