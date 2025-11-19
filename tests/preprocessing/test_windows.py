from datetime import datetime

from src.deforestation_predictor.preprocessing.windows import get_input_window_range

def test_get_input_window_range():
    target_date = datetime(2023, 10, 1)

    # Explicit example: context=3, gap=1 → [T-3, T-1]
    start, end = get_input_window_range(target_date, context=3, gap=1)
    assert start == datetime(2023, 7, 1)
    assert end == datetime(2023, 9, 1)

    # Default example: should be same as context=3, gap=1
    start, end = get_input_window_range(target_date)
    assert start == datetime(2023, 7, 1)
    assert end == datetime(2023, 9, 1)