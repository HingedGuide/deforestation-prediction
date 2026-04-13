"""
Script Name: Monthly Evaluation Analysis

Description:
    This standalone script evaluates a trained deep learning model on the test
    dataset and calculates performance metrics (F0.5, Precision, Recall) broken
    down by month. It works natively with MultiTile datasets.
    
    The script dynamically handles both 2D pre-aggregated data (4D tensors) and 
    3D temporal sequences (5D tensors), allowing for fair baseline comparisons 
    without requiring code changes.
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

# --- Project Imports ---
from deforestation_predictor.models.architectures import (
    ResUNet, ResUNet3D, ConvLSTM3D, ViViTSegmentation
)
from deforestation_predictor.training.train_experiment import MultiTileDataset, get_coordinates

# ==============================================================================
# Evaluation Logic
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

def evaluate_per_month(model, loader, device, model_type, mode, threshold):
    """
    Evaluates the model on the test set and calculates metrics per month.
    Dynamically handles 4D inputs (pre-aggregated 2D data) and 5D inputs (temporal 3D data).
    Assumes the 'month' variable is located at channel index 5.
    """
    model.eval()
    month_dict = {m: {'preds': [], 'targets': []} for m in range(1, 13)}

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Testing per Month"):
            X, y = X.to(device), y.to(device)

            # Check tensor dimensions to dynamically handle both 2D (aggregated) and 3D (temporal) inputs
            if X.dim() == 4:
                # 2D Model Case: Data is pre-aggregated and sequence length is 1
                month_vals = X[:, 5, 0, 0]
                X_input = X
                
            elif X.dim() == 5:
                # 3D Model Case: Data contains a sequence of time steps
                if mode != 'sequence':
                    raise ValueError("5D input requires mode='sequence' to access historical steps.")
                
                # Extract the month value from the last timestep
                month_vals = X[:, 5, -1, 0, 0]
                
                # Reshape for 2D Early Fusion models expecting flattened time, or keep 5D for 3D models
                b, c, t, h, w = X.shape
                X_input = X.view(b, c * t, h, w) if model_type == 'resunet' else X
                
            else:
                raise ValueError(f"Unexpected input dimension from DataLoader: {X.dim()}")

            # Reverse the preprocessing normalization: round((val / 255) * 12)
            months = torch.round((month_vals / 255.0) * 12.0).int().cpu().numpy()

            # Forward pass
            logits = model(X_input)
            if logits.shape[1] == 1:
                logits = logits.squeeze(1)

            probs = torch.sigmoid(logits)
            valid_pixels = (y == 0) | (y == 1)

            for i in range(X.shape[0]):
                vp = valid_pixels[i]
                if vp.sum() > 0:
                    m = int(months[i])
                    if 1 <= m <= 12:
                        month_dict[m]['preds'].append(probs[i][vp].cpu().numpy())
                        month_dict[m]['targets'].append(y[i][vp].cpu().numpy())

    results = []
    for m in range(1, 13):
        if not month_dict[m]['targets']:
            continue

        y_true = np.concatenate(month_dict[m]['targets'])
        y_scores = np.concatenate(month_dict[m]['preds'])
        y_pred = (y_scores >= threshold).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

        results.append({
            'Month': m,
            'F0.5': f05,
            'Precision': prec,
            'Recall': rec,
            'Valid_Pixels': len(y_true)
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('Month').reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance per month.")
    
    # Data Paths
    parser.add_argument('--image_path', type=str, default='./laura_preprocessing/output', help="Root folder for .npy files")
    parser.add_argument('--tiles', type=str, default="00N_000E", help="Comma-separated list of tile IDs")
    parser.add_argument('--country', type=str, default="Laos", help="Country name used in the TIFF mask filename")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    
    # Model Configuration
    parser.add_argument("--model_type", type=str, required=True, choices=["resunet", "resunet3d", "convlstm3d", "vivit"])
    parser.add_argument("--mode", type=str, default="sequence", choices=["sequence", "snapshot"])
    parser.add_argument("--context_months", type=int, default=12, help="Sequence length for the input data")
    
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

    # 3. Determine input shapes dynamically
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

    # 5. Evaluate
    print(f"Starting monthly evaluation (Threshold: {args.threshold})...")
    monthly_df = evaluate_per_month(model, test_loader, device, args.model_type, args.mode, args.threshold)

    # 6. Output Results
    print("\n=== Monthly Performance ===")
    print(monthly_df.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"monthly_results_{args.model_type}_{args.context_months}m.csv")
    monthly_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()