from datetime import datetime
import pandas as pd
from pandas import DateOffset


CONTEXT_MONTHS = 12  # number of months in the input window
GAP_MONTHS = 1       # number of months gap between input and target. I changed this to 1 because I no longer use
                     # the aggregated monthly inputs (e.g. lastthreemonths)


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