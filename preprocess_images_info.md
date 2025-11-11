# Preprocessing for 3D CNN Deforestation Model

This module prepares `.tif` files and ground-truth rasters so they can be used as input for a **3D CNN**.

## File naming convention

All rasters follow:

`tileLat_tileLon_YYYY-MM-DD_variable.tif`

Example:  
`00N_010E_2023-08-15_wetlands.tif`

From this we extract:
- `tile_id`  → `00N_010E`
- `date`     → `2023-08-15`
- `variable` → `wetlands`

## Core functions

- **`parse_filename()`**  
  Parses a filename into: `tile_id`, `date`, `variable`, `path`.

- **`build_raster_catalog(data_root)`**  
  Scans `data_root` for `.tif` files and builds a `DataFrame` with:
  `tile_id`, `date`, `variable`, `path`.

- **`build_gt_catalog(gt_root)`**  
  Same idea, but for ground-truth files (e.g. `*_gt.tif`).

- **`get_records_for_dates(catalog, tile_id, start, end)`**  
  Returns all rasters for one `tile_id` between `start` and `end` dates (inclusive).

- **`stack_rasters(records)`**  
  Loads those rasters and stacks them into a numpy array of shape:  
  `[V, T, H, W]` = variables × time steps × height × width.

- **`get_input_window_range(target_date, context, gap)`**  
  Given a target month `T`, returns the input window `[start, end]`.  
  For the current setup: `context=12`, `gap=3` → input = `[T−14, T−3]`.

- **`build_sample(tile_id, target_date, catalog, gt_catalog, context, gap)`**  
  Main helper. For one `tile_id` and one `target_date` it returns:
  - `X`: input cube `[V, T, H, W]`
  - `y`: ground truth `[H, W]`
  - `meta`: info about variables and dates

## Usage idea

1. Build catalogs:
   - `catalog = build_raster_catalog(input_root)`
   - `gt_catalog = build_gt_catalog(gt_root)`
2. For each `(tile_id, target_date)`:
   - `X, y, meta = build_sample(...)`
3. Feed `X` to a 3D CNN as `(channels=V, depth=T, height=H, width=W)`.

Tests (`pytest`) create fake `.tif` files and check that each function behaves as expected.
