from datetime import datetime
import pandas as pd
from pandas import DateOffset

# Default constants for the temporal window
CONTEXT_MONTHS = 12
GAP_MONTHS = 1


def get_input_window_range(
        target_date: datetime | pd.Timestamp,
        context: int = CONTEXT_MONTHS,
        gap: int = GAP_MONTHS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Given a target_date T, return (start, end) for the input window.

    The default values ensure that the preprocessing pipeline builds
    a standard window that can be further sliced during training.
    """
    T = pd.to_datetime(target_date)
    end = T - DateOffset(months=gap)
    start = end - DateOffset(months=context - 1)
    return start, end