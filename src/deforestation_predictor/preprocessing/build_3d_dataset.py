from __future__ import annotations

from pathlib import Path
from pandas import DateOffset
import numpy as np
import pandas as pd
import logging
import sys

from deforestation_predictor.preprocessing.catalog import (
    build_raster_catalog,
    build_gt_catalog,
    compute_variable_maxima,
)
from deforestation_predictor.preprocessing.splits import (
    build_target_table,
    filter_targets_with_full_window,
    split_targets_by_time,
    TemporalSplitConfig,
)
from deforestation_predictor.preprocessing.builder import (
    build_sample,
    balanced_random_spatial_crops
)
from deforestation_predictor.preprocessing.windows import (
    CONTEXT_MONTHS,
    GAP_MONTHS,
)
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

# RAW (per-tile) roots that you already have
RAW_INPUT_ROOT = PROJECT_ROOT / "data" / "input"
RAW_GT_ROOT = PROJECT_ROOT / "data" / "groundtruth"

# Region settings
REGION_ID = "GABON"

# Bounding box for Gabon
GABON_BOUNDS = (8.4, -4.1, 14.6, 2.3)  # (west, south, east, north)

# The 4 tiles that surround Gabon in your tiling scheme
REGION_TILES = [
    "00N_000E",
    "10N_000E",
    "00N_010E",
    "10N_010E",
]

# Where mosaiced + clipped tifs will be written
REGION_INPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "input"
REGION_GT_ROOT = PROJECT_ROOT / "data" / "processed" / "groundtruth"

# Where 3D dataset .npz will be written
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed_3d" / REGION_ID

# Which variables you want the 3D CNN to see (snapshot / monthly)
# firealerts, nightlights and fw are not available for full time range
MONTHLY_VARS = [
    #'firealerts',
    #'nightlights',
    'precipitation',
    'temperature',
    'confidence',
    #'fwi',
    'lastmonth',
    'timesinceloss',
    'totallossalerts',
    'previoussameseason',
    'patchdensity'
]

STATIC_VARS = [
    "elevation",
    "slope",
    "peatlands",
    "initialforestcover",
    "historicloss",
    "forestheight",
    "forestedgedensity",
    "landpercentage",
    "populationcurrent",
    "populationincrease",
    "closenesstoroads",
    "closenesstowaterways",
    "closenesstocropland",
    "closenesstococoa",
    "closenesstocoffee",
    "closenesstofiber",
    "closenesstosoybean",
    "closenesstocattleabove2000",
    "closenesstocattleabove10000",
    "closenesstomining",
    "palmoilmills",
    "croplandcapacity100p",
    "croplandcapacitybelow50p",
    "croplandcapacityover50p",
    "catexcap",
    "aridityannual",
    "ariditydriestquarter",
    "closenesstoforestedge",
    "wetlands",
    "wdpa",
]

# Categorical variables should not be scaled by maxima
CATEGORICAL_VARS: set[str] = {
    'wetlands',
    'wdpa',
}

CONTEXT = CONTEXT_MONTHS
ANCHOR_DATE = "2022-01-01"
MAX_CONTEXT = 12
GAP = 6  # Prediction horizon

SPLIT_CFG = TemporalSplitConfig(
    train_end="2022-12-01",  # 12 months of training
    val_end="2023-12-01",    # 12 months of validation (contiguous)
    test_start="2024-07-01", # 6 months gap (Jan-Jun 2024 skipped)
    test_end="2025-06-01",   # 12 months of testing
    context=MAX_CONTEXT,
    gap=GAP,
)

# Validation start date (real, after filtering)
# First few months of validation are filtered out to prevent leakage from training period
VAL_START_DATE_REAL = "2023-07-01"

PATCH_SIZE = 64

# Number of patches to extract per sample
# I put this quite high to have more data for training
# and also to have some overlapping patches so that the model can learn from it more robustly

PATCHES_PER_SAMPLE_TRAIN = 800
PATCHES_PER_SAMPLE_VAL = 800
PATCHES_PER_SAMPLE_TEST = 800

# Fraction of patches that should contain deforestation (y=1)
POS_FRACTION_TRAIN = 0.5

# Threshold for forest mask
FOREST_MASK_THRESHOLD = 2000.0


# ------------- MAIN PIPELINE ------------- #

def main():
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = output_root / "preprocessing.log"
    logger = setup_logger(log_path)

    logger.info("Starting pipeline execution...")
    logger.info(f"Output directory set as: {output_root}")

    # 0) Mosaic + clip to Gabon
    logger.info("[0] Preparing mosaiced & clipped rasters for region GABON...")
    mosaic_and_clip_region(
        raw_input_root=RAW_INPUT_ROOT,
        raw_gt_root=RAW_GT_ROOT,
        out_input_root=REGION_INPUT_ROOT,
        out_gt_root=REGION_GT_ROOT,
        region_id=REGION_ID,
        bounds=GABON_BOUNDS,
        tile_ids=REGION_TILES,
    )

    # From here on we work ONLY with the mosaiced/clipped region
    INPUT_ROOT = REGION_INPUT_ROOT / REGION_ID
    GT_ROOT = REGION_GT_ROOT / REGION_ID

    # 1) Build catalogs
    logger.info("[1] Building input catalog (unfiltered)...")
    full_catalog = build_raster_catalog(str(INPUT_ROOT))

    # --- FIX 1: Corrected logging syntax ---
    logger.info(f"Raw catalog rows: {len(full_catalog)}")
    logger.info(f"Raw unique variables: {sorted(full_catalog['variable'].unique())}")

    # Dynamic (monthly) catalog
    catalog = full_catalog[full_catalog["variable"].isin(MONTHLY_VARS)].reset_index(drop=True)

    # --- FIX 2: Corrected logging syntax ---
    logger.info(f"Filtered catalog rows (monthly): {len(catalog)}")
    logger.info(f"Filtered unique monthly variables: {sorted(catalog['variable'].unique())}")

    # Static catalog
    static_catalog = full_catalog[full_catalog["variable"].isin(STATIC_VARS)].reset_index(drop=True)

    # Forest mask catalog
    forestmask_catalog = (
        full_catalog[full_catalog["variable"] == "initialforestcover"]
        .reset_index(drop=True)
    )
    # --- FIX 3: Corrected logging syntax ---
    logger.info(f"Forest mask records: {len(forestmask_catalog)}")

    logger.info("[2] Building GT catalog...")
    gt_catalog = build_gt_catalog(str(GT_ROOT))
    logger.info(f"    -> {len(gt_catalog)} GT records")

    # --- FIX 4: Stop if no Ground Truth found ---
    if gt_catalog.empty:
        logger.error("CRITICAL: No Ground Truth records found. Stopping to prevent crash.")
        logger.error(f"Please check if GT files exist in: {GT_ROOT}")
        return

    # 3) Build target table from GT and enforce Anchor Date
    logger.info(f"[3] Building target table (starting from {ANCHOR_DATE})...")
    targets = build_target_table(gt_catalog, min_date=ANCHOR_DATE)

    # 4) Filter targets with FULL windows based on MAX_CONTEXT
    # This ensures consistency: all models will use the exact same pixels/dates.
    logger.info(f"[4] Filtering targets using MAX_CONTEXT={MAX_CONTEXT}...")
    targets_full = filter_targets_with_full_window(
        targets,
        catalog,
        context=MAX_CONTEXT,
        gap=GAP,
    )

    # 5) Split into train / val / test using the new logic
    logger.info("[5] Splitting targets into train/val/test...")
    train_targets, val_targets_raw, test_targets = split_targets_by_time(
        targets_full,
        SPLIT_CFG,
    )

    # Create a gap between train and val by removing early val targets
    valid_start_real = pd.to_datetime(VAL_START_DATE_REAL)
    val_targets = val_targets_raw[val_targets_raw["date"] >= valid_start_real].reset_index(drop=True)

    logger.info(f"    -> Filtered out gap (Jan-Jun 2023) from validation.")
    # --------------------------------------------------------

    logger.info(f"    train: {len(train_targets)}")
    logger.info(f"    val:   {len(val_targets)}")
    logger.info(f"    test:  {len(test_targets)}")

    # Save split metadata as CSV
    splits_dir = output_root / "splits"
    splits_dir.mkdir(exist_ok=True)
    train_targets.to_csv(splits_dir / "train_targets.csv", index=False)
    val_targets.to_csv(splits_dir / "val_targets.csv", index=False)
    test_targets.to_csv(splits_dir / "test_targets.csv", index=False)

    # 6) Compute maxima *only from training time range*
    logger.info("[6] Computing per-variable maxima from TRAIN catalog only...")

    train_end = pd.to_datetime(SPLIT_CFG.train_end)
    max_input_date = train_end - DateOffset(months=GAP)
    min_target_date = train_targets["date"].min()

    if pd.isna(min_target_date):
        logger.warning("No training targets found, cannot compute maxima reliably.")
        maxima = {}  # handle appropriately or return
    else:
        min_input_date = min_target_date - DateOffset(months=CONTEXT + GAP - 1)

        train_catalog_for_max = full_catalog[
            (full_catalog["date"] >= min_input_date)
            & (full_catalog["date"] <= max_input_date)
            & (full_catalog["variable"].isin(MONTHLY_VARS + STATIC_VARS))
            ].reset_index(drop=True)

        logger.info(
            f"    -> using {len(train_catalog_for_max)} rasters for maxima "
            f"from {min_input_date.date()} to {max_input_date.date()}"
        )
        maxima = compute_variable_maxima(train_catalog_for_max)

    # 7) Materialize samples for each split
    logger.info("[7] Building and saving samples...")

    for split_name, split_targets in [
        ("train", train_targets),
        ("val", val_targets),
        ("test", test_targets)
    ]:
        if split_targets.empty:
            logger.info(f"Skipping {split_name} split (no targets).")
            continue

        build_and_save_split(
            split_name=split_name,
            targets=split_targets,
            catalog=catalog,
            gt_catalog=gt_catalog,
            maxima=maxima,
            output_root=output_root,
            static_catalog=static_catalog,
            forestmask_catalog=forestmask_catalog,
            forest_mask_threshold=FOREST_MASK_THRESHOLD,
            categorical_vars=CATEGORICAL_VARS,
            logger=logger
        )

    logger.info("[DONE] All samples saved.")


def build_and_save_split(
        split_name: str,
        targets: pd.DataFrame,
        catalog: pd.DataFrame,
        gt_catalog: pd.DataFrame,
        maxima: dict[str, float],
        output_root: Path,
        static_catalog: pd.DataFrame | None = None,
        forestmask_catalog: pd.DataFrame | None = None,
        forest_mask_threshold: float = 0.0,
        categorical_vars: set[str] | None = None,
        logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)

    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for i, row in enumerate(targets.itertuples(index=False)):
        tile_id = row.tile_id
        target_date = row.date

        try:
            X, y, meta = build_sample(
                tile_id=tile_id,
                target_date=target_date,
                catalog=catalog,
                gt_catalog=gt_catalog,
                context=CONTEXT,
                gap=GAP,
                static_catalog=static_catalog,
                normalize=True,
                overflow="clip",
                nan_policy="zero",
                cast="float32",
                maxima=maxima,
                forestmask_catalog=forestmask_catalog,
                forest_mask_threshold=forest_mask_threshold,
                categorical_vars=categorical_vars,
            )
        except Exception as e:
            logger.warning(
                f"Failed to build sample for {tile_id} @ {target_date}: {e}"
            )
            continue

        if split_name == "train":
            n_patches = PATCHES_PER_SAMPLE_TRAIN
            X_patches, y_patches = balanced_random_spatial_crops(
                X, y, patch_size=PATCH_SIZE, n_patches=n_patches, pos_fraction=POS_FRACTION_TRAIN,
            )
        else:
            n_patches = PATCHES_PER_SAMPLE_VAL if split_name == "val" else PATCHES_PER_SAMPLE_TEST
            from deforestation_predictor.preprocessing.builder import random_spatial_crops
            X_patches, y_patches = random_spatial_crops(
                X, y, patch_size=PATCH_SIZE, n_patches=n_patches,
            )

        for j in range(n_patches):
            fname = f"{tile_id}_{target_date.date()}_patch{j:03d}.npz"
            out_path = split_dir / fname

            np.savez_compressed(
                out_path,
                X=X_patches[j],
                y=y_patches[j],
                tile_id=tile_id,
                target_date=str(target_date.date()),
                patch_id=j,
                variables=np.array(meta["variables"], dtype=object),
                dates=np.array([d.isoformat() for d in meta["dates"]], dtype=object),
            )

            records.append({
                "tile_id": tile_id,
                "target_date": target_date,
                "patch_id": j,
                "path": str(out_path),
            })

        if (i + 1) % 10 == 0:
            logger.info(f"    [{split_name}] processed {i + 1}/{len(targets)} samples")

    if records:
        index_df = pd.DataFrame(records)
        index_df.to_csv(output_root / f"{split_name}_index.csv", index=False)
        logger.info(f"    [{split_name}] saved {len(records)} samples and {split_name}_index.csv")


if __name__ == "__main__":
    main()