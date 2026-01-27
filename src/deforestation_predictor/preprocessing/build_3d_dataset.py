from __future__ import annotations

import json
from pathlib import Path
from pandas import DateOffset
import numpy as np
import pandas as pd
import logging
import sys
import rasterio

from deforestation_predictor.preprocessing.catalog import (
    build_raster_catalog,
    build_gt_catalog,
    compute_variable_maxima,
)
from deforestation_predictor.preprocessing.splits import TemporalSplitConfig
from deforestation_predictor.preprocessing.mosaic_and_clip_bbox import (
    mosaic_and_clip_region,
)


# Configuring logging
def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ------------- CONFIG ------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# RAW (per-tile) roots
RAW_INPUT_ROOT = PROJECT_ROOT / "data" / "input"
RAW_GT_ROOT = PROJECT_ROOT / "data" / "groundtruth"

REGION_ID = "GABON"
GABON_BOUNDS = (8.4, -4.1, 14.6, 2.3)
REGION_TILES = ["00N_000E", "10N_000E", "00N_010E", "10N_010E"]

REGION_INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "input"
REGION_GT_ROOT = PROJECT_ROOT / "data" / "processed" / "groundtruth"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / REGION_ID

# Variables
MONTHLY_VARS = [
    'precipitation', 'temperature', 'confidence', 'lastmonth',
    'lastthreemonths', 'lastsixmonths', 'timesinceloss',
    'totallossalerts', 'previoussameseason', 'patchdensity',
    'smoothedsixmonths', 'smoothedtotal',
]

STATIC_VARS = [
    "elevation", "slope", "peatlands", "initialforestcover", "historicloss",
    "forestheight", "forestedgedensity", "landpercentage", "populationcurrent",
    "populationincrease", "closenesstoroads", "closenesstowaterways",
    "closenesstocropland", "closenesstococoa", "closenesstocoffee",
    "closenesstofiber", "closenesstosoybean", "closenesstocattleabove2000",
    "closenesstocattleabove10000", "closenesstomining", "palmoilmills",
    "croplandcapacity100p", "croplandcapacitybelow50p", "croplandcapacityover50p",
    "catexcap", "aridityannual", "ariditydriestquarter", "closenesstoforestedge",
    "wetlands", "wdpa",
]

CATEGORICAL_VARS: set[str] = {'wetlands', 'wdpa'}

# Pipeline Params
FOREST_MASK_THRESHOLD = 2000.0
ANCHOR_DATE = "2021-01-01"
GAP = 1
CONTEXT = 1 # Not used for building the cube, but for split calculations

SPLIT_CFG = TemporalSplitConfig(
    train_end="2023-04-01",
    val_end="2024-03-01",
    test_start="2024-04-01",
    test_end="2024-09-01",
    context=CONTEXT,
    gap=GAP,
)


def main():
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root / "preprocessing.log")

    logger.info("Starting On-the-Fly Dataset Generation...")

    # 0) Mosaic + clip
    mosaic_and_clip_region(
        raw_input_root=RAW_INPUT_ROOT,
        raw_gt_root=RAW_GT_ROOT,
        out_input_root=REGION_INPUT_ROOT,
        out_gt_root=REGION_GT_ROOT,
        region_id=REGION_ID,
        bounds=GABON_BOUNDS,
        tile_ids=REGION_TILES,
    )

    INPUT_ROOT = REGION_INPUT_ROOT / REGION_ID
    GT_ROOT = REGION_GT_ROOT / REGION_ID

    # 1) Build catalogs
    full_catalog = build_raster_catalog(str(INPUT_ROOT))
    catalog = full_catalog[full_catalog["variable"].isin(MONTHLY_VARS)].reset_index(drop=True)
    static_catalog = full_catalog[full_catalog["variable"].isin(STATIC_VARS)].reset_index(drop=True)
    forestmask_catalog = full_catalog[full_catalog["variable"] == "initialforestcover"].reset_index(drop=True)

    gt_catalog = build_gt_catalog(str(GT_ROOT))
    if gt_catalog.empty:
        logger.error("No Ground Truth found.")
        return

    # 2) Compute Maxima (Train Only)
    logger.info("[2] Computing maxima...")
    train_end = pd.to_datetime(SPLIT_CFG.train_end)
    train_mask = (full_catalog["date"] < train_end) | (full_catalog["variable"].isin(STATIC_VARS))
    train_catalog = full_catalog[train_mask].reset_index(drop=True)
    maxima = compute_variable_maxima(train_catalog)
    
    with open(output_root / "maxima.json", "w") as f:
        json.dump(maxima, f)

    # 3) Build Dense 4D Array
    logger.info("[3] Building dense 4D arrays (mmap)...")
    build_dense_region(
        catalog=catalog,
        static_catalog=static_catalog,
        gt_catalog=gt_catalog,
        maxima=maxima,
        output_root=output_root,
        logger=logger,
        anchor_date=ANCHOR_DATE,
        forestmask_path=forestmask_catalog.iloc[0]["path"] if not forestmask_catalog.empty else None
    )


def build_dense_region(catalog, static_catalog, gt_catalog, maxima, output_root, logger, anchor_date, forestmask_path):
    """
    Constructs one massive features.npy [C, T, H, W] and labels.npy [T, H, W]
    and saves positive indices for balancing.
    """
    # 1. Determine timeline
    dates = sorted(list(set(catalog["date"].dt.date) | set(gt_catalog["date"].dt.date)))
    dates = [d for d in dates if d >= pd.to_datetime(anchor_date).date()]
    dates_str = [str(d) for d in dates]
    
    T = len(dates)
    logger.info(f"Timeline: {T} months from {dates[0]} to {dates[-1]}")

    # 2. Determine Spatial Shape from first file
    ref_path = catalog.iloc[0]["path"]
    with rasterio.open(ref_path) as src:
        H, W = src.shape
        profile = src.profile
    
    C = len(MONTHLY_VARS) + len(STATIC_VARS)
    logger.info(f"Dimensions: [C={C}, T={T}, H={H}, W={W}]")

    # 3. Create MMAP files
    feat_path = output_root / "features.npy"
    label_path = output_root / "labels.npy"

    # Initialize with zeros
    X_mmap = np.lib.format.open_memmap(feat_path, mode='w+', dtype='float32', shape=(C, T, H, W))
    y_mmap = np.lib.format.open_memmap(label_path, mode='w+', dtype='uint8', shape=(T, H, W))

    # 4. Load Static Data (Once)
    # We append static vars to the end of the channel dimension
    # Static data is repeated across time in the Dataset __getitem__, NOT here to save space?
    # Actually, user requested [C, T, H, W] input. 
    # To optimize, we usually keep static separate, but the prompt asks for "large features.npy".
    # We will replicate static vars into the MMAP for simplicity, or we can save them separately.
    # PROMPT SPECIFIC: "The output tensor must have the shape: [Channels, Sequence_Length, Height, Width]."
    # To support mmap slicing efficiently, we will write static vars into the 4D array
    # BUT broadcasting static vars 36 times is wasteful. 
    # DECISION: We will write static vars as the LAST channels for EVERY time step. 
    # Wait, strict shape [C, T, H, W] means C channels, T timesteps.
    # If variables are static, they don't change over T. 
    # Let's save static separately? No, the user wants ONE dataset file usually. 
    # Reference `multicropdataset.py` merges them. 
    # Strategy: We will build X_mmap as [C, T, H, W]. Yes, it repeats static data. Disk is cheap-ish.

    static_layers = []
    for var in STATIC_VARS:
        row = static_catalog[static_catalog["variable"] == var]
        if not row.empty:
            with rasterio.open(row.iloc[0]["path"]) as src:
                arr = src.read(1).astype('float32')
                # Normalize
                if var not in CATEGORICAL_VARS and var in maxima:
                    arr /= (maxima[var] if maxima[var] > 0 else 1.0)
                # NaNs to 0
                arr = np.nan_to_num(arr, nan=0.0)
                static_layers.append(arr)
        else:
            static_layers.append(np.zeros((H, W), dtype='float32'))
    
    static_block = np.stack(static_layers) # [C_stat, H, W]

    # 5. Fill Time Steps
    positive_indices = {} # t_idx -> list of [y, x]

    for t_idx, date in enumerate(dates):
        # A. Labels
        gt_row = gt_catalog[gt_catalog["date"].dt.date == date]
        if not gt_row.empty:
            with rasterio.open(gt_row.iloc[0]["path"]) as src:
                lbl = src.read(1)
                # Mask: 0=No, 1=Deforestation. 
                # (Assuming GT is already clean, or use forest mask to set ignore regions)
                # User's code used to set 2 as ignore. Let's keep it simple: 1 is positive.
                lbl = lbl.astype('uint8')
                y_mmap[t_idx] = lbl
                
                # Store positives for balancing
                # Use np.argwhere. This might be large, but usually deforestation is sparse.
                pos_locs = np.argwhere(lbl == 1)
                if len(pos_locs) > 0:
                    positive_indices[t_idx] = pos_locs.tolist() # [[y,x], ...]
        else:
            # If no GT for this month, fill 0 (or ignore?)
            # Assuming 0.
            pass

        # B. Dynamic Features
        for c_idx, var in enumerate(MONTHLY_VARS):
            row = catalog[(catalog["date"].dt.date == date) & (catalog["variable"] == var)]
            if not row.empty:
                with rasterio.open(row.iloc[0]["path"]) as src:
                    arr = src.read(1).astype('float32')
                    if var in maxima:
                        arr /= (maxima[var] if maxima[var] > 0 else 1.0)
                    arr = np.nan_to_num(arr, nan=0.0)
                    X_mmap[c_idx, t_idx] = arr
            else:
                pass # Leave as 0.0

        # C. Static Features (Broadcast)
        # Static vars start after monthly vars
        start_c = len(MONTHLY_VARS)
        X_mmap[start_c:, t_idx] = static_block

        if (t_idx + 1) % 5 == 0:
            logger.info(f"Processed {t_idx + 1}/{T} months")

    # Flush changes to disk
    X_mmap.flush()
    y_mmap.flush()

    # 6. Save Metadata
    meta = {
        "dates": dates_str,
        "variables": MONTHLY_VARS + STATIC_VARS,
        "positive_indices": positive_indices, # dict[str(t), list[list[y,x]]]
        "shape": [C, T, H, W]
    }
    
    # Save indices separately if too large for JSON, but usually okay for sparse masks
    # If huge, use npy. Let's use npy for indices to be safe.
    # We flatten the list: [t, y, x]
    all_positives = []
    for t, locs in positive_indices.items():
        for yx in locs:
            all_positives.append([t, yx[0], yx[1]])
    
    np.save(output_root / "positive_indices.npy", np.array(all_positives, dtype='int32'))
    
    # Remove large list from json
    del meta["positive_indices"]
    
    with open(output_root / "stats.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Dataset generation complete.")

if __name__ == "__main__":
    main()