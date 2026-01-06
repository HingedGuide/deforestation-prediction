from datetime import datetime
import pandas as pd
from pandas import DateOffset


def get_input_window_range(
        target_date: datetime | pd.Timestamp,
        context: int,  # No default value anymore
        gap: int,  # No default value anymore
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Given a target_date T, return (start, end) for the input window.

    This function requires explicit context and gap to ensure
    comparability between different model experiments.
    """
    T = pd.to_datetime(target_date)
    end = T - DateOffset(months=gap)
    start = end - DateOffset(months=context - 1)
    return start, end