"""
Script to predict multiple full tiles using a sliding window approach with deep learning models.
Loads processed .npy arrays and saves the output as binarized GeoTIFFs for ArcGIS inspection.
"""

import argparse
import torch
import numpy as np
import rasterio
import os
from tqdm import tqdm

from deforestation_predictor.models.architectures import (
    ResUNet, ResUNet3D, ViViTSegmentation, ConvLSTM3D
)

def get_model(model_type, in_channels, time_depth, device):
    """
    Factory function to initialize the correct architecture.
    """
    if model_type == "resunet":
        return ResUNet(in_channels=in_channels, num_classes=1).to(device)
    elif model_type == "resunet3d":
        return ResUNet3D(in_channels=in_channels, time_depth=time_depth, num_classes=1).to(device)
    elif model_type == "convlstm3d":
        return ConvLSTM3D(in_channels=in_channels, time_depth=time_depth, num_classes=1).to(device)
    elif model_type == "vivit":
        return ViViTSegmentation(in_channels=in_channels, time_depth=time_depth, num_classes=1).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def predict_sliding_window(model, X, model_type, mode, patch_size=256, overlap=32, device='cuda'):
    """
    Runs inference using a sliding window to prevent memory issues on large spatial tiles.
    Returns the continuous probability map.
    """
    C, T, H, W = X.shape
    output_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    model.eval()
    stride = patch_size - overlap
    
    print(f"Running inference on shape {X.shape} with patches of {patch_size}x{patch_size}...")

    for r in tqdm(range(0, H - patch_size + 1, stride), leave=False):
        for c in range(0, W - patch_size + 1, stride):
            # Extract spatial patch and copy to make it writable (fixes UserWarning)
            patch = X[:, :, r:r+patch_size, c:c+patch_size].copy()
            
            # Convert to tensor, cast to float, add batch dim, move to device (fixes RuntimeError)
            inp = torch.from_numpy(patch).float().unsqueeze(0).to(device)
            
            # Flatten time dimension for 2D ResUNet in sequence mode
            if model_type == 'resunet' and mode == 'sequence':
                b, ch, t, h_dim, w_dim = inp.shape
                inp = inp.view(b, ch * t, h_dim, w_dim)
                
            with torch.no_grad():
                logits = model(inp)
                if logits.shape[1] == 1: 
                    logits = logits.squeeze(1)
                
                # Use sigmoid for binary classification
                probs = torch.sigmoid(logits)
                
            pred_patch = probs.cpu().numpy()[0]
            
            # Add to output map
            output_map[r:r+patch_size, c:c+patch_size] += pred_patch
            count_map[r:r+patch_size, c:c+patch_size] += 1.0
            
    # Average the overlapping areas to smooth the predictions
    count_map[count_map == 0] = 1
    output_map /= count_map
    
    return output_map

def main():
    parser = argparse.ArgumentParser(description="Predict Multiple Tiles with DL Models and Output Binary Maps")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the directory containing .npy files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pth model weights")
    parser.add_argument("--model_type", type=str, required=True, choices=["resunet", "resunet3d", "convlstm3d", "vivit"])
    parser.add_argument("--mode", type=str, default="sequence", choices=["sequence", "snapshot"])
    parser.add_argument("--tiles", type=str, default="00N_000E", help="Comma-separated list of tile IDs")
    parser.add_argument("--country", type=str, default="Laos", help="Country name for locating the TIFF mask")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which temporal split to use")
    parser.add_argument("--context_months", type=int, default=12, help="Sequence length for input")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save the GeoTIFFs")
    
    # Argument for the binary threshold
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold to binarize the output map")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert the comma-separated string into a list
    tile_list = args.tiles.split(",")
    
    # Loop over all provided tiles
    for tile_id in tile_list:
        print(f"\n--- Processing Tile: {tile_id} ---")
        
        # Define file paths
        img_file = os.path.join(args.data_root, f"{tile_id}_var_{args.split}.npy")
        mask_file = os.path.join(args.data_root, f"{tile_id}_mask_{args.country}.tiff")
        output_path = os.path.join(args.output_dir, f"prediction_{tile_id}.tif")
        
        if not os.path.exists(img_file):
            print(f"Warning: Cannot find data file for {tile_id}: {img_file}. Skipping...")
            continue
        if not os.path.exists(mask_file):
            print(f"Warning: Cannot find reference mask TIFF for {tile_id}: {mask_file}. Skipping...")
            continue
            
        print(f"Loading data from {img_file}...")
        var_array = np.load(img_file, mmap_mode='r')
        
        # Slice the temporal dimension to match the expected context length
        total_time = var_array.shape[1]
        seq_len = args.context_months if args.mode == 'sequence' else 1
        
        if total_time < seq_len:
            print(f"Warning: Data only contains {total_time} timesteps, but {seq_len} are requested. Skipping {tile_id}...")
            continue
            
        X = var_array[:, -seq_len:, :, :]
        print(f"Input tensor shape after temporal slicing: {X.shape}")
        
        # Model Initialization (Done once per tile to ensure correct dimensions if they vary, though usually static)
        base_channels = X.shape[0]
        in_channels = base_channels * seq_len if (args.mode == 'sequence' and args.model_type == 'resunet') else base_channels
        
        print(f"Initializing {args.model_type} model...")
        model = get_model(args.model_type, in_channels, seq_len, device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
        # Execute inference to get probabilities
        prob_map = predict_sliding_window(model, X, args.model_type, args.mode, device=device)
        
        # Apply the threshold to create a binary mask (1 for deforested, 0 for forested)
        print(f"Applying threshold of {args.threshold} to create a binary map...")
        binary_map = (prob_map >= args.threshold).astype(np.uint8)
        
        # Print the exact number of predicted pixels for validation
        num_def = np.sum(binary_map)
        print(f"-> Success! Found {num_def} deforested pixels (1s) out of {binary_map.size} total pixels.")
        
        # Export as GeoTIFF using the geospatial profile from the mask
        print(f"Saving binary prediction to {output_path}...")
        with rasterio.open(mask_file) as src:
            profile = src.profile.copy()
            
        # Update profile to uint8 for binary data, and set nodata=0 so ArcGIS makes it transparent!
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0)
        
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(binary_map, 1)
            
    print("\nAll requested tiles have been processed.")

if __name__ == "__main__":
    main()