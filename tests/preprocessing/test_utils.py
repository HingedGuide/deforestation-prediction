from datetime import datetime

from deforestation_predictor.utils.filenames import parse_filename


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