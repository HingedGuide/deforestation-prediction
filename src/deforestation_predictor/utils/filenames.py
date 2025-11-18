from pathlib import Path
from datetime import datetime


def parse_filename(filename: str | Path) -> dict:
    """
    Parse a filename into components.

    Supported patterns:

    1) Tiled format (original behaviour):
         'coord1_coord2_YYYY-MM-DD_variable.tif'
         Example:
             '00N_000E_2004-01-01_wetlands.tif'
         -> tile_id = '00N_000E'

    2) Region format (for mosaiced regions like GABON):
         'regionId_YYYY-MM-DD_variable.tif'
         Example:
             'GABON_2004-01-01_wetlands.tif'
         -> tile_id = 'GABON'

    In both cases, `variable` may contain underscores.

    Returns a dictionary with:
        - tile_id (str)
        - date (datetime)
        - variable (str)
        - path (str)
    """
    filename = Path(filename)
    name = filename.stem  # remove extension e.g. '.tif'
    parts = name.split("_")

    if len(parts) < 3:
        # Too few parts to match any supported pattern
        raise ValueError(f"Unexpected filename format: {filename.name}")

    # Try pattern 1: coord1_coord2_YYYY-MM-DD_variable...
    # Here the date should be at index 2.
    if len(parts) >= 4:
        coord1 = parts[0]
        coord2 = parts[1]
        date_str_tile = parts[2]

        try:
            date = datetime.strptime(date_str_tile, "%Y-%m-%d")
        except ValueError:
            date = None
        else:
            variable = "_".join(parts[3:])
            tile_id = f"{coord1}_{coord2}"
            return {
                "tile_id": tile_id,
                "date": date,
                "variable": variable,
                "path": str(filename),
            }

    # If that didn't work, try pattern 2: regionId_YYYY-MM-DD_variable...
    # Here the date should be at index 1.
    date_str_region = parts[1]
    try:
        date = datetime.strptime(date_str_region, "%Y-%m-%d")
    except ValueError as e:
        # Not tile pattern, not region pattern → fail
        raise ValueError(f"Unexpected filename format: {filename.name}") from e

    tile_id = parts[0]
    variable = "_".join(parts[2:])

    return {
        "tile_id": tile_id,
        "date": date,
        "variable": variable,
        "path": str(filename),
    }
