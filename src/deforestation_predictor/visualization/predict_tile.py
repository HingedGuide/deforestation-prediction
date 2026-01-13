import argparse
import torch
import numpy as np
import rasterio
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Project imports
from deforestation_predictor.preprocessing.catalog import build_raster_catalog, get_records_for_dates
from deforestation_predictor.preprocessing.builder import stack_rasters, normalize_cube_auto
from deforestation_predictor.preprocessing.windows import get_input_window_range
from deforestation_predictor.models.architectures import (
    ResUNet, ResUNet3D, ViViTSegmentation, ConvLSTM3D
)

# Hardcoded variable lists (ensure these match training!)
MONTHLY_VARS = [
    'precipitation', 'temperature', 'confidence', 'lastmonth', 
    'timesinceloss', 'totallossalerts', 'previoussameseason', 'patchdensity'
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

def get_model(model_type, in_channels, time_depth, device):
    # Same factory function as before
    if model_type == "resunet":
        return ResUNet(in_channels, time_depth).to(device)
    elif model_type == "resunet3d":
        return ResUNet3D(in_channels, time_depth).to(device)
    elif model_type == "convlstm3d":
        return ConvLSTM3D(in_channels, time_depth).to(device)
    elif model_type == "vivit":
        return ViViTSegmentation(in_channels, time_depth).to(device)
    else:
        raise ValueError(f"Model {model_type} not supported in this script yet.")

def build_inference_cube(tile_id, target_date, catalog, static_catalog, context, gap, maxima):
    """Loads data and builds a normalized input cube [C, T, H, W]."""
    print(f"Building input cube for {tile_id} @ {target_date}...")
    
    # 1. Temporal Window
    start, end = get_input_window_range(target_date, context=context, gap=gap)
    subset = get_records_for_dates(catalog, tile_id, start, end)
    
    if subset.empty:
        raise ValueError(f"No data found for {tile_id} in window {start}-{end}")

    # 2. Stack Dynamic
    X, variables, dates = stack_rasters(subset) # [V, T, H, W]
    
    # Retrieve geospatial profile from the first raster (for later export)
    ref_path = subset.iloc[0]["path"]
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
    
    # 3. Normalize Dynamic
    X = normalize_cube_auto(X, variables, maxima=maxima, overflow="clip", nan_policy="zero")
    
    # 4. Add Static (Broadcast & Normalize)
    if static_catalog is not None and not static_catalog.empty:
        static_records = static_catalog[static_catalog["tile_id"] == tile_id]
        static_arrays = []
        found_static_vars = []
        
        for v in sorted(static_records["variable"].unique()):
            # Take most recent snapshot
            path = static_records[static_records["variable"] == v].sort_values("date").iloc[-1]["path"]
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
            static_arrays.append(arr)
            found_static_vars.append(v)
            
        if static_arrays:
            static_cube = np.stack(static_arrays, axis=0) # [V_stat, H, W]
            
            # Normalize static (trick: make temporarily 4D)
            static_cube_4d = static_cube[:, None, :, :]
            static_cube_4d = normalize_cube_auto(
                static_cube_4d, found_static_vars, maxima=maxima, overflow="clip", nan_policy="zero"
            )
            static_cube = static_cube_4d[:, 0, :, :]
            
            # Broadcast over Time
            T_len = X.shape[1]
            static_broadcast = np.repeat(static_cube[:, None, :, :], T_len, axis=1)
            
            # Concatenate
            X = np.concatenate([X, static_broadcast], axis=0)

    return X, profile

def predict_sliding_window(model, X, patch_size=256, overlap=32, device='cuda'):
    """
    Runs inference using a sliding window to prevent memory issues.
    X shape: [C, T, H, W]
    """
    C, T, H, W = X.shape
    output_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    model.eval()
    stride = patch_size - overlap
    
    print(f"Running inference on shape {X.shape} with patches...")

    # Pad image if needed (simple: we skip edge cases or crop for now, 
    # for production you would add padding)
    
    for r in tqdm(range(0, H - patch_size + 1, stride)):
        for c in range(0, W - patch_size + 1, stride):
            # Extract patch
            patch = X[:, :, r:r+patch_size, c:c+patch_size]
            
            # To Tensor [1, C, T, H, W]
            inp = torch.from_numpy(patch).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(inp)
                probs = torch.softmax(logits, dim=1)[:, 1, :, :] # Class 1 probability
                
            pred_patch = probs.cpu().numpy()[0]
            
            # Add to map
            output_map[r:r+patch_size, c:c+patch_size] += pred_patch
            count_map[r:r+patch_size, c:c+patch_size] += 1.0
            
    # Average overlapping areas
    # Avoid division by zero
    count_map[count_map == 0] = 1
    output_map /= count_map
    
    return output_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to processed data root (e.g. data/processed/input)")
    parser.add_argument("--maxima_file", type=str, required=True, help="Path to maxima.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--tile_id", type=str, default="GABON")
    parser.add_argument("--date", type=str, default="2024-01-01", help="Target date YYYY-MM-DD")
    parser.add_argument("--context_months", type=int, default=12)
    parser.add_argument("--output_path", type=str, default="prediction.tif")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Catalogs
    print("Loading catalogs...")
    full_cat = build_raster_catalog(args.data_root)
    
    # Filter catalogs
    dyn_cat = full_cat[full_cat["variable"].isin(MONTHLY_VARS)].reset_index(drop=True)
    stat_cat = full_cat[full_cat["variable"].isin(STATIC_VARS)].reset_index(drop=True)
    
    # Load Maxima
    with open(args.maxima_file, "r") as f:
        maxima = json.load(f)

    # 2. Build Input
    target_date = pd.to_datetime(args.date)
    X, profile = build_inference_cube(
        args.tile_id, target_date, dyn_cat, stat_cat, 
        args.context_months, gap=1, maxima=maxima
    )
    
    # 3. Load Model
    in_channels = X.shape[0]
    time_depth = X.shape[1]
    model = get_model(args.model_type, in_channels, time_depth, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # 4. Predict
    prob_map = predict_sliding_window(model, X, device=device)
    
    # 5. Save as GeoTIFF
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    
    print(f"Saving to {args.output_path}...")
    with rasterio.open(args.output_path, "w", **profile) as dst:
        dst.write(prob_map, 1)

if __name__ == "__main__":
    main()