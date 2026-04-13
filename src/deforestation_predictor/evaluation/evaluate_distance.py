"""
Script Name: Spatial Distance Evaluation Analysis

Description:
    This standalone script evaluates a trained deep learning model on the test
    dataset and calculates performance metrics (F0.5, Precision, Recall) broken
    down by the pixel's distance to recent past deforestation.
    
    Distance categories:
    - 0: Within past deforestation
    - 1: Adjacent to past deforestation (<= 400m)
    - 2: > 400m and <= 800m
    - 3: > 800m
    
    It assumes the 'lastsixmonths' variable is located at channel index 0.
"""

import argparse
import os
import random
import pandas as pd
import numpy as np
import torch
import rasterio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, fbeta_score
from scipy.ndimage import distance_transform_cdt

# --- Project Imports ---
from deforestation_predictor.models.architectures import (
    ResUNet, ResUNet3D, ConvLSTM3D, ViViTSegmentation
)
from deforestation_predictor.training.train_experiment import MultiTileDataset, get_coordinates


# ==============================================================================
# 2. Evaluation Logic
# ==============================================================================

def get_model(model_type, in_channels, time_depth, device):
    """
    Factory function to initialize the model architecture.
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

def evaluate_spatial_distance(model, loader, device, model_type, mode, threshold, past_def_channel=0):
    """
    Evaluates the model on the test set and calculates metrics per distance category.
    Uses Chebyshev (chessboard) distance to match the 400m pixel grid logic.
    """
    model.eval()
    
    # Define the 4 distance categories
    cat_dict = {
        0: {'preds': [], 'targets': [], 'name': 'Within past def. (0m)'},
        1: {'preds': [], 'targets': [], 'name': 'Adjacent (<= 400m)'},
        2: {'preds': [], 'targets': [], 'name': '> 400m and <= 800m'},
        3: {'preds': [], 'targets': [], 'name': '> 800m'}
    }

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Testing Spatial Distance"):
            X, y = X.to(device), y.to(device)

            if mode == 'sequence':
                past_def_vals = torch.sum(X[:, past_def_channel, :, :, :], dim=1).cpu().numpy()
                
                b, c, t, h, w = X.shape
                X_input = X.view(b, c * t, h, w) if model_type == 'resunet' else X
            else:
                past_def_vals = X[:, past_def_channel, :, :].cpu().numpy()
                X_input = X

            logits = model(X_input)
            if logits.shape[1] == 1:
                logits = logits.squeeze(1)

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = y.cpu().numpy()

            for i in range(X.shape[0]):
                # Create a binary mask of past deforestation (values > 0)
                past_def_mask = past_def_vals[i] > 0
                
                # Invert mask for distance_transform (it calculates distance to False pixels)
                bg_mask = ~past_def_mask
                
                # Calculate chessboard distance
                if not bg_mask.any():
                    # Entire patch is deforested
                    dist_map = np.zeros_like(past_def_mask, dtype=int)
                elif bg_mask.all():
                    # No past deforestation in the patch
                    dist_map = np.full_like(past_def_mask, 999, dtype=int)
                else:
                    dist_map = distance_transform_cdt(bg_mask, metric='chessboard')
                
                # Cap distances at 3 (everything > 800m falls into category 3)
                dist_map_cat = np.where(dist_map >= 3, 3, dist_map)
                
                valid_pixels = (targets[i] == 0) | (targets[i] == 1)
                
                # Append predictions to their respective categories
                for cat in range(4):
                    mask_cat = (dist_map_cat == cat) & valid_pixels
                    if mask_cat.any():
                        cat_dict[cat]['preds'].append(probs[i][mask_cat])
                        cat_dict[cat]['targets'].append(targets[i][mask_cat])

    # Calculate metrics
    results = []
    for cat, data in cat_dict.items():
        if not data['targets']:
            continue

        y_true = np.concatenate(data['targets'])
        y_scores = np.concatenate(data['preds'])
        y_pred = (y_scores >= threshold).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

        results.append({
            'Distance Category': data['name'],
            'F0.5': f05,
            'Precision': prec,
            'Recall': rec,
            'Valid_Pixels': len(y_true)
        })

    df = pd.DataFrame(results)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance per spatial distance.")
    
    # Data Paths
    parser.add_argument('--image_path', type=str, default='./laura_preprocessing/output', help="Root folder for .npy files")
    parser.add_argument('--tiles', type=str, default="00N_000E", help="Comma-separated list of tile IDs")
    parser.add_argument('--country', type=str, default="Laos", help="Country name used in the TIFF mask filename")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    
    # Model Configuration
    parser.add_argument("--model_type", type=str, required=True, choices=["resunet", "resunet3d", "convlstm3d", "vivit"])
    parser.add_argument("--mode", type=str, default="sequence", choices=["sequence", "snapshot"])
    parser.add_argument("--context_months", type=int, default=12, help="Sequence length for 3D models")
    parser.add_argument("--past_def_channel", type=int, default=0, help="Channel index for past deforestation (default: 0 for lastsixmonths)")
    
    # Testing Configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--test_samples", type=int, default=50000, help="Max samples for testing phase to prevent OOM")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold locked from validation")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the resulting CSV")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Generate File Paths
    tile_list = args.tiles.split(",")
    test_img_paths = [os.path.join(args.image_path, f'{t}_var_test.npy') for t in tile_list]
    test_gt_paths = [os.path.join(args.image_path, f'{t}_gt_test.npy') for t in tile_list]
    mask_paths = [os.path.join(args.image_path, f'{t}_mask_{args.country}.tiff') for t in tile_list]

    seq_len = args.context_months if args.mode == 'sequence' else 1

    # 2. Load Dataset
    print("Generating Test Coordinates...")
    test_coords = get_coordinates(test_gt_paths, mask_paths, 64, seq_len, balance=0, max_samples=args.test_samples)
    
    test_ds = MultiTileDataset(test_img_paths, test_gt_paths, test_coords, 64, seq_len, epoch_size=None)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if len(test_ds) == 0:
        print("No samples found in the test set. Exiting.")
        return

    # 3. Determine input shapes
    sample_data = np.load(test_img_paths[0], mmap_mode='r')
    base_channels = sample_data.shape[0]

    if args.mode == 'sequence' and args.model_type == 'resunet':
        in_channels = base_channels * seq_len
    else:
        in_channels = base_channels

    print(f"Detected Shape -> Base Channels: {base_channels}, In Channels: {in_channels}")

    # 4. Initialize Model
    model = get_model(args.model_type, in_channels, seq_len, device)
    print(f"Loading weights from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 5. Evaluate Spatial Distance
    print(f"Starting spatial distance evaluation (Threshold: {args.threshold})...")
    distance_df = evaluate_spatial_distance(
        model, test_loader, device, args.model_type, args.mode, 
        threshold=args.threshold, past_def_channel=args.past_def_channel
    )

    # 6. Output Results
    print("\n=== Distance Performance ===")
    print(distance_df.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"distance_results_{args.model_type}_{args.context_months}m.csv")
    distance_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()