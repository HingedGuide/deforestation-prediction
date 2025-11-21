from __future__ import annotations

from pathlib import Path
from pandas import DateOffset
import numpy as np
import pandas as pd

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
MONTHLY_VARS = [
    'firealerts',
    'nightlights',
    'precipitation',
    'temperature',
    'confidence',
    'fwi',
    'lastmonth',
    'timesinceloss',
    'totallossalerts',
    'previoussameseason',
    'patchdensity'
    # extend as needed
]

STATIC_VARS = [
    "elevation",
    "slope",
    "wetlands",                 # if present; otherwise remove
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
    "wdpa",
    "catexcap",
    "aridityannual",
    "ariditydriestquarter",
    "closenesstoforestedge",
    # easy to add: "cattlesmoothed", "dpicoal", "dpiconvgas", ...
]

CONTEXT = CONTEXT_MONTHS
GAP = GAP_MONTHS

SPLIT_CFG = TemporalSplitConfig(
    train_end="2023-06-30",
    val_end="2023-12-31",
    context=CONTEXT,
    gap=GAP,
)

PATCH_SIZE = 64

PATCHES_PER_SAMPLE_TRAIN = 64   # e.g. 64 patches per (tile_id, date)
PATCHES_PER_SAMPLE_VAL = 16     # fewer for val
PATCHES_PER_SAMPLE_TEST = 16    # fewer for test

# Fraction of patches that should contain deforestation (y=1)
POS_FRACTION_TRAIN = 0.5        # roughly 50% positive, 50% negative

# Threshold for forest mask (how many underlying pixels were forest)
FOREST_MASK_THRESHOLD = 2000.0 # out of 10000

# ------------- MAIN PIPELINE ------------- #

def main():
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    # 0) Mosaic + clip to Gabon (idempotent enough to just run each time)
    print("[0] Preparing mosaiced & clipped rasters for region GABON...")
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
    print("[1] Building input catalog (unfiltered)...")
    full_catalog = build_raster_catalog(str(INPUT_ROOT))

    print("Raw catalog rows:", len(full_catalog))
    print("Raw unique variables:", sorted(full_catalog["variable"].unique()))

    # Dynamic (monthly) catalog: only the vars we want as time series
    catalog = full_catalog[full_catalog["variable"].isin(MONTHLY_VARS)].reset_index(drop=True)

    print("Filtered catalog rows (monthly):", len(catalog))
    print("Filtered unique monthly variables:", sorted(catalog["variable"].unique()))

    # Static catalog: vars we want to broadcast across time
    static_catalog = full_catalog[full_catalog["variable"].isin(STATIC_VARS)].reset_index(drop=True)

    # forest mask catalog based on initialforestcover
    forestmask_catalog = (
        full_catalog[full_catalog["variable"] == "initialforestcover"]
        .reset_index(drop=True)
    )
    print("Forest mask records:", len(forestmask_catalog))

    print("[2] Building GT catalog...")
    gt_catalog = build_gt_catalog(str(GT_ROOT))
    print(f"    -> {len(gt_catalog)} GT records")

    # 3) Build target table from GT
    print("[3] Building target table from GT...")
    targets = build_target_table(gt_catalog)
    print(f"    -> {len(targets)} (tile_id, date) pairs")

    # 4) Filter targets to those with a full temporal window
    print("[4] Filtering targets with full windows...")
    targets_full = filter_targets_with_full_window(
        targets,
        catalog,
        context=CONTEXT,
        gap=GAP,
    )
    print(f"    -> {len(targets_full)} valid targets after window check")

    # 5) Split into train / val / test
    print("[5] Splitting targets into train/val/test...")
    train_targets, val_targets, test_targets = split_targets_by_time(
        targets_full,
        SPLIT_CFG,
    )
    print(f"    train: {len(train_targets)}")
    print(f"    val:   {len(val_targets)}")
    print(f"    test:  {len(test_targets)}")

    # Save split metadata as CSV
    splits_dir = output_root / "splits"
    splits_dir.mkdir(exist_ok=True)
    train_targets.to_csv(splits_dir / "train_targets.csv", index=False)
    val_targets.to_csv(splits_dir / "val_targets.csv", index=False)
    test_targets.to_csv(splits_dir / "test_targets.csv", index=False)

    # 6) Compute maxima *only from training time range* to avoid leakage
    print("[6] Computing per-variable maxima from TRAIN catalog only...")

    train_end = pd.to_datetime(SPLIT_CFG.train_end)

    # Latest month that can appear in a train input window
    max_input_date = train_end - DateOffset(months=GAP)

    # Earliest month that can appear in a train input window
    min_target_date = train_targets["date"].min()
    min_input_date = min_target_date - DateOffset(months=CONTEXT + GAP - 1)

    train_catalog_for_max = full_catalog[
        (full_catalog["date"] >= min_input_date)
        & (full_catalog["date"] <= max_input_date)
        & (full_catalog["variable"].isin(MONTHLY_VARS + STATIC_VARS))
    ].reset_index(drop=True)

    print(
        f"    -> using {len(train_catalog_for_max)} rasters for maxima "
        f"from {min_input_date.date()} to {max_input_date.date()}"
    )

    maxima = compute_variable_maxima(train_catalog_for_max)


    # 7) Materialize samples for each split
    print("[7] Building and saving samples...")

    build_and_save_split(
        split_name="train",
        targets=train_targets,
        catalog=catalog,
        gt_catalog=gt_catalog,
        maxima=maxima,
        output_root=output_root,
        static_catalog=static_catalog,
        forestmask_catalog=forestmask_catalog,
        forest_mask_threshold=FOREST_MASK_THRESHOLD,
    )

    build_and_save_split(
        split_name="val",
        targets=val_targets,
        catalog=catalog,
        gt_catalog=gt_catalog,
        maxima=maxima,
        output_root=output_root,
        static_catalog=static_catalog,
        forestmask_catalog=forestmask_catalog,
        forest_mask_threshold=FOREST_MASK_THRESHOLD,
    )

    build_and_save_split(
        split_name="test",
        targets=test_targets,
        catalog=catalog,
        gt_catalog=gt_catalog,
        maxima=maxima,
        output_root=output_root,
        static_catalog=static_catalog,
        forestmask_catalog=forestmask_catalog,
        forest_mask_threshold=FOREST_MASK_THRESHOLD,
    )

    print("[DONE] All samples saved.")


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
):
    """
    Loop over all (tile_id, date) targets in a split, build samples,
    and save them as .npz files under:
        OUTPUT_ROOT / split_name / f"{tile_id}_{date}.npz"
    """
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
            )
        except Exception as e:
            print(
                f"[Warning] Failed to build sample for "
                f"{tile_id} @ {target_date}: {e}"
            )
            continue

        # decide #patches per split
        if split_name == "train":
            n_patches = PATCHES_PER_SAMPLE_TRAIN
        elif split_name == "val":
            n_patches = PATCHES_PER_SAMPLE_VAL
        else:  # "test"
            n_patches = PATCHES_PER_SAMPLE_TEST

        if split_name == "train":
            # TRAIN: class-balanced patches
            X_patches, y_patches = balanced_random_spatial_crops(
                X,
                y,
                patch_size=PATCH_SIZE,
                n_patches=n_patches,
                pos_fraction=POS_FRACTION_TRAIN,
            )
        else:
            # VAL/TEST: unbiased random patches that mirror real prevalence
            from deforestation_predictor.preprocessing.builder import (
                random_spatial_crops,
            )

            X_patches, y_patches = random_spatial_crops(
                X,
                y,
                patch_size=PATCH_SIZE,
                n_patches=n_patches,
            )

        # save each patch as its own .npz
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
                dates=np.array(
                    [d.isoformat() for d in meta["dates"]],
                    dtype=object,
                ),
            )

            records.append(
                {
                    "tile_id": tile_id,
                    "target_date": target_date,
                    "patch_id": j,
                    "path": str(out_path),
                }
            )

        if (i + 1) % 50 == 0:
            print(f"    [{split_name}] processed {i+1}/{len(targets)} samples")

    if records:
        index_df = pd.DataFrame(records)
        index_df.to_csv(output_root / f"{split_name}_index.csv", index=False)
        print(
            f"    [{split_name}] saved {len(records)} samples and "
            f"{split_name}_index.csv"
        )


if __name__ == "__main__":
    main()
