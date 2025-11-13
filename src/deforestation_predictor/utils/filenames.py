from pathlib import Path
from datetime import datetime


def parse_filename(filename: str | Path) -> dict:
    """
    Parse a filename formatted as:
        'coordinate1_coordinate2_YYYY-MM-DD_variable.tif'

    Returns a dictionary with:
        - tile_id (str)
        - date (datetime)
        - variable (str)
        - path (str)
    """
    filename = Path(filename)
    name = filename.stem  # remove '.tif'
    parts = name.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename.name}")

    coord1 = parts[0]
    coord2 = parts[1]
    date_str = parts[2]
    variable = "_".join(parts[3:])  # supports variables with underscores

    tile_id = f"{coord1}_{coord2}"
    date = datetime.strptime(date_str, "%Y-%m-%d")

    return {
        "tile_id": tile_id,
        "date": date,
        "variable": variable,
        "path": str(filename),
    }