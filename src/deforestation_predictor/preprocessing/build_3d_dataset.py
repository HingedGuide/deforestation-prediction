from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from deforestation_predictor.preprocessing.catalog import (
    build_raster_catalog,
    build_gt_catalog,
)
from deforestation_predictor.preprocessing.splits import (
    build_target_table,
    filter_targets_with_full_window,
    split_targets_by_time,
    TemporalSplitConfig,
)
from deforestation_predictor.preprocessing.builder import build_sample
from deforestation_predictor.preprocessing.windows import (
    CONTEXT_MONTHS,
    GAP_MONTHS,
)
from deforestation_predictor.preprocessing.catalog import compute_variable_maxima


# ------------- CONFIG ------------- #

# Root folders (adjust to your setup)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_ROOT = PROJECT_ROOT / "data" / "input" / "00N_000E"                   # monthly variables
GT_ROOT = PROJECT_ROOT / "data" / "groundtruth" / "00N_000E"                # ground-truth tiles
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed_3d" / "00N_000E"           # where .npz files will be saved


# Which variables you want the 3D CNN to see (snapshot / monthly)
MONTHLY_VARS = [
    "lastmonth",
    "nightlights",
    "precipitation",
    "firealerts"
    # extend as needed
]

# Temporal settings
CONTEXT = CONTEXT_MONTHS
GAP = GAP_MONTHS

# Split config
SPLIT_CFG = TemporalSplitConfig(
    train_end="2023-06-30",
    val_end="2023-12-31",
    context=CONTEXT,
    gap=GAP,
)


# ------------- MAIN PIPELINE ------------- #

def main():
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1) Build catalogs
    print("[1] Building input catalog (unfiltered)...")
    catalog = build_raster_catalog(str(INPUT_ROOT))

    print("Raw catalog rows:", len(catalog))
    print("Raw unique variables:", sorted(catalog["variable"].unique()))

    # 💡 Keep ONLY the monthly variables we want for the 3D CNN
    catalog = catalog[catalog["variable"].isin(MONTHLY_VARS)].reset_index(drop=True)

    print("Filtered catalog rows:", len(catalog))
    print("Filtered unique variables:", sorted(catalog["variable"].unique()))
    print(f"    -> {len(catalog)} input records, "
          f"{catalog['variable'].nunique()} variables "
          f"across {catalog['tile_id'].nunique()} tiles")

    print("[2] Building GT catalog...")
    gt_catalog = build_gt_catalog(str(GT_ROOT))
    print(f"    -> {len(gt_catalog)} GT records")

    # Optional: precompute maxima once for speed
    print("[3] Computing per-variable maxima for normalization...")
    maxima = compute_variable_maxima(catalog)

    # 2) Build target table from GT
    print("[4] Building target table from GT...")
    targets = build_target_table(
        gt_catalog,
        # optionally clamp training horizon here:
        # min_date="2021-01-01",
        # max_date="2024-12-31",
    )
    print(f"    -> {len(targets)} (tile_id, date) pairs")

    # 3) Filter targets to those with a full window
    print("[5] Filtering targets with full windows...")
    targets_full = filter_targets_with_full_window(
        targets,
        catalog,
        context=CONTEXT,
        gap=GAP,
    )
    print(f"    -> {len(targets_full)} valid targets after window check")

    # 4) Split into train / val / test
    print("[6] Splitting targets into train/val/test...")
    train_targets, val_targets, test_targets = split_targets_by_time(
        targets_full,
        SPLIT_CFG,
    )
    print(f"    train: {len(train_targets)}")
    print(f"    val:   {len(val_targets)}")
    print(f"    test:  {len(test_targets)}")

    # Save split metadata as CSV for reference
    splits_dir = output_root / "splits"
    splits_dir.mkdir(exist_ok=True)
    train_targets.to_csv(splits_dir / "train_targets.csv", index=False)
    val_targets.to_csv(splits_dir / "val_targets.csv", index=False)
    test_targets.to_csv(splits_dir / "test_targets.csv", index=False)

    # 5) Materialize samples for each split
    print("[7] Building and saving samples...")

    build_and_save_split(
        split_name="train",
        targets=train_targets,
        catalog=catalog,
        gt_catalog=gt_catalog,
        maxima=maxima,
        output_root=output_root,
    )

    build_and_save_split(
        split_name="val",
        targets=val_targets,
        catalog=catalog,
        gt_catalog=gt_catalog,
        maxima=maxima,
        output_root=output_root,
    )

    build_and_save_split(
        split_name="test",
        targets=test_targets,
        catalog=catalog,
        gt_catalog=gt_catalog,
        maxima=maxima,
        output_root=output_root,
    )

    print("[DONE] All samples saved.")


def build_and_save_split(
    split_name: str,
    targets: pd.DataFrame,
    catalog: pd.DataFrame,
    gt_catalog: pd.DataFrame,
    maxima: dict[str, float],
    output_root: Path,
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

        # You can skip try/except if your targets are already guaranteed valid
        try:
            X, y, meta = build_sample(
                tile_id=tile_id,
                target_date=target_date,
                catalog=catalog,
                gt_catalog=gt_catalog,
                context=CONTEXT,
                gap=GAP,
                normalize=True,
                overflow="clip",
                nan_policy="zero",
                cast=None,   # keep float32; or "float16" or "uint8"
                use_cache=True,
                maxima=maxima,  # use global maxima; avoids recompute
            )
        except Exception as e:
            print(f"[Warning] Failed to build sample for "
                  f"{tile_id} @ {target_date}: {e}")
            continue

        # Filename: tileid_YYYY-MM-DD.npz
        fname = f"{tile_id}_{target_date.date()}.npz"
        out_path = split_dir / fname

        # Save compressed
        np.savez_compressed(
            out_path,
            X=X,
            y=y,
            tile_id=tile_id,
            target_date=str(target_date.date()),
            variables=np.array(meta["variables"], dtype=object),
            dates=np.array([d.isoformat() for d in meta["dates"]], dtype=object),
        )

        records.append(
            {
                "tile_id": tile_id,
                "target_date": target_date,
                "path": str(out_path),
            }
        )

        if (i + 1) % 50 == 0:
            print(f"    [{split_name}] processed {i+1}/{len(targets)} samples")

    # Save index CSV for the split (so Dataset can just read this)
    if records:
        index_df = pd.DataFrame(records)
        index_df.to_csv(output_root / f"{split_name}_index.csv", index=False)
        print(f"    [{split_name}] saved {len(records)} samples and "
              f"{split_name}_index.csv")


if __name__ == "__main__":
    main()
