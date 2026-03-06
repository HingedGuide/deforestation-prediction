# Deforestation 3D Dataset Builder

A modular preprocessing pipeline for converting raw input rasters into patch-based spatio-temporal datasets suitable for 3D CNNs, ConvLSTMs, and transformer models for deforestation prediction.

---

## Overview

This repository contains all components required to:

- Mosaic and clip tile-based rasters into a coherent study region  
- Build catalogs for dynamic, static, and ground-truth rasters  
- Create temporal input windows (context + gap)  
- Filter to only targets with full input windows  
- Split targets into train/validation/test by date  
- Compute normalization maxima only from training data  
- Generate full spatio-temporal data cubes `[V, T, H, W]`  
- Apply forest masks (non-forest → ignore label = `2`)  
- Extract balanced patches for training  
- Extract unbiased patches for validation/test  
- Save every patch as compressed `.npz`

The full end-to-end pipeline is implemented in:
`deforestation_predictor.preprocessing.build_3d_dataset`

---

## Directory Structure
````
preprocessing/
├── build_3d_dataset.py         # Main pipeline script
├── builder.py                  # Sample building, stacking, normalization, cropping
├── catalogs.py                 # Catalog creation and metadata parsing
├── legacy_ff.py                # 2D dataset builder (for reference)
├── mosaic_and_clip_bbox.py     # Mosaic and clip rasters to study area
├── splits.py                   # Target creation + temporal splits
└── windows.py                  # Temporal window creation and filtering
````
---

## Pipeline Steps
### 1. Mosaic and Clip Region
Tiles are merged and clipped into a single regional dataset.

Output filename is as follows:
`{REGION}_{YYYY}-{MM}-{DD}_{Variable}.tif`


This ensures compatibility with the filename parser and catalog system.

### 2. Build Raster & Ground-Truth Catalogs

Each catalog contains:

| Column     | Meaning                                 |
|------------|------------------------------------------|
| tile_id    | Original tile name                       |
| date       | Timestamp (monthly)                      |
| variable   | Variable name (e.g. `firealerts`)        |
| path       | Absolute `.tif` path                     |

### 3. Temporal Input Windows

For a target date `T`, the input window is:

Start: `T - (context + gap)`  
End: `T - gap - 1`

Where `context` is the number of months of input data, and `gap` is the number of months between the last input and the target date.


### 4. Full-Window Filtering

A (tile_id, date) pair is kept only if:

- all variables requested exist for all months in the input window  
- no missing dates or missing variables

### 5. Train/Validation/Test Split

Time-based splitting:

- **Train:** `date <= train_end`
- **Validation:** `train_end < date <= val_end`
- **Test:** `date > val_end`

### 6. Training-Only Normalization

Each variable is divided by its **maximum value computed solely from the training input window**.

This prevents information leakage.

### 7. Static Variables

Static layers (e.g. elevation, slope, forest height):

1. Loaded once per tile  
2. Normalized using the same maxima dictionary  
3. Broadcast across the time dimension  
4. Concatenated to the dynamic channels

### 8. Forest Mask

`initialforestcover` is used to create a binary forest mask:

- forest → keep  
- non-forest/water → set GT to ignore label `2`

### 9. Patch Extraction

#### Training (balanced)
Uses:
``balanced_random_spatial_crops`` to enforce a chosen fraction of patches containing positive pixels.

#### Validation/Test (unbiased)
Uses: ``random_spatial_crops`` to extract patches without bias and to preserve real world deforestation frequency.

Each patch is saved as a compressed `.npz` with filename as
```tileid_YYYY_MM_DD_patchNNN.npz```

Each ```.npz``` file file contains:

| Key           | Shape / Type         | Description                      |
|---------------|----------------------|----------------------------------|
| `X`           | `[V, T, H, W]`       | Input cube                       |
| `y`           | `[H, W]`             | Target mask (0, 1, 2=ignore)     |
| `variables`   | list                 | Channel names in order           |
| `dates`       | list                 | Months in the input window       |
| `tile_id`     | str                  | Original tile id                 |
| `target_date` | str/date             | Date of prediction               |
| `patch_id`    | int                  | Patch index                      |

example file is ``GABON_2023-04-01_patch005.npz``

---

## Running the Pipeline

### 1. Organize Raw Data

````
data/
├── input/<tile_id>/filename.tif            # Raw input rasters
└── groundtruth/<tile_id>/filename.tif      # Raw ground-truth rasters
````
Filenames must follow the convention:
`{tile_id}_{YYYY}-{MM}-{DD}_{Variable}.tif`

### 2. Configure `build_3d_dataset.py`

Set:

- Region ID  
- Bounding box  
- Tiles  
- Dynamic & static variable lists  
- Forest mask threshold  
- Split dates  
- Patch size & counts  

### 3. Run

``python build_3d_dataset.py``

Outputs will be saved to the specified output directory:
``data/processed/{region_id}/3d_dataset/``
Each split will have its own subdirectory:
``train/``, ``val/``, ``test/``.

---

## License

MIT License.

---

## Acknowledgements

Built using preprocessing components developed for the WUR–WWF deforestation research framework.
