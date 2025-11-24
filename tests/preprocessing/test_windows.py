from datetime import datetime
from src.deforestation_predictor.preprocessing.windows import get_input_window_range

def test_get_input_window_range():
    target_date = datetime(2023, 10, 1)

    # Explicit example: context=3, gap=1 → months [Jul, Aug, Sep]
    start, end = get_input_window_range(target_date, context=3, gap=1)
    assert start == datetime(2023, 7, 1)
    assert end == datetime(2023, 9, 1)

    # Default example: context=12, gap=1 → months [Oct 2022 ... Sep 2023]
    start, end = get_input_window_range(target_date)
    assert start == datetime(2022, 10, 1)
    assert end == datetime(2023, 9, 1)
