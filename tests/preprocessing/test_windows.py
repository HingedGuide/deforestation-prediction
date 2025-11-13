from datetime import datetime

from src.deforestation_predictor.preprocessing.windows import get_input_window_range

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