"""
Script Name: Emerging Frontiers Evaluation Analysis

Description:
    This standalone script evaluates a trained deep learning model on the test
    dataset and calculates performance metrics specifically for "emerging frontiers".
    It does this by isolating pixels that have NOT experienced any deforestation
    in the preceding X months, effectively filtering out the expansion of already 
    active deforestation fronts.
    
    It dynamically creates a historical deforestation mask by stacking and summing
    the 'lastmonth' variable over a specified time window.
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import rasterio
from torch.utils.data import DataLoader
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

def evaluate_new_frontiers(model, loader, device, model_type, mode, threshold, lastmonth_channel, lookback_months):
    """
    Evaluates the model on new deforestation frontiers by masking out areas
    that were deforested in the recent past (lookback_months).
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    total_valid_pixels = 0
    total_frontier_pixels = 0

    with torch.no_grad():
        for X, y in tqdm(loader, desc=f"Testing Emerging Frontiers ({lookback_months}m lookback)"):
            X, y = X.to(device), y.to(device)

            # We need temporal sequence data to look back in time
            if mode != 'sequence':
                raise ValueError("This analysis requires mode='sequence' to access historical 'lastmonth' steps.")
            
            # 1. Isolate the 'lastmonth' channel for the last X months
            # Shape of X: [Batch, Channels, Time, Height, Width]
            historical_stack = X[:, lastmonth_channel, -lookback_months:, :, :]
            
            # 2. Sum over the time dimension to find any recent deforestation
            # Shape of recent_def_sum: [Batch, Height, Width]
            recent_def_sum = torch.sum(historical_stack, dim=1)
            
            # 3. Create boolean masks
            # True if there WAS deforestation in the lookback window
            had_recent_def = recent_def_sum > 0 
            # True if it is a strictly new frontier
            is_new_frontier = ~had_recent_def
            
            # 4. Handle model input reshaping (e.g., for 2D ResUNet Early Fusion)
            b, c, t, h, w = X.shape
            X_input = X.view(b, c * t, h, w) if model_type == 'resunet' else X

            # 5. Forward pass
            logits = model(X_input)
            if logits.shape[1] == 1:
                logits = logits.squeeze(1)

            probs = torch.sigmoid(logits)
            
            # 6. Combine masks to get the final pixels for evaluation
            # Only evaluate pixels that are valid (0 or 1) AND are new frontiers
            valid_pixels = (y == 0) | (y == 1)
            eval_mask = valid_pixels & is_new_frontier
            
            # Track statistics for reporting
            total_valid_pixels += valid_pixels.sum().item()
            total_frontier_pixels += eval_mask.sum().item()

            for i in range(b):
                mask_i = eval_mask[i]
                if mask_i.sum() > 0:
                    all_preds.append(probs[i][mask_i].cpu().numpy())
                    all_targets.append(y[i][mask_i].cpu().numpy())

    # Calculate final metrics
    if not all_targets:
        print("Warning: No new frontier pixels found in the test set to evaluate.")
        return pd.DataFrame()

    y_true = np.concatenate(all_targets)
    y_scores = np.concatenate(all_preds)
    y_pred = (y_scores >= threshold).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)

    # Compile the results
    results = [{
        'Evaluation Category': f'New Frontiers (>{lookback_months}m)',
        'F0.5': f05,
        'Precision': prec,
        'Recall': rec,
        'Evaluated_Pixels': len(y_true),
        'Total_Original_Valid_Pixels': total_valid_pixels,
        'Percentage_Retained': f"{(len(y_true) / total_valid_pixels * 100):.2f}%"
    }]

    df = pd.DataFrame(results)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate model specifically on emerging deforestation frontiers.")
    
    # Data Paths
    parser.add_argument('--image_path', type=str, default='./laura_preprocessing/output', help="Root folder for .npy files")
    parser.add_argument('--tiles', type=str, default="00N_000E", help="Comma-separated list of tile IDs")
    parser.add_argument('--country', type=str, default="Laos", help="Country name used in the TIFF mask filename")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    
    # Model Configuration
    parser.add_argument("--model_type", type=str, required=True, choices=["resunet", "resunet3d", "convlstm3d", "vivit"])
    parser.add_argument("--mode", type=str, default="sequence", choices=["sequence", "snapshot"])
    parser.add_argument("--context_months", type=int, default=6, help="Sequence length for 3D models")
    
    # Specific Frontier Arguments
    parser.add_argument("--lastmonth_channel", type=int, default=0, help="Channel index of the 'lastmonth' variable (default: 3)")
    parser.add_argument("--lookback_months", type=int, default=6, help="Number of months to check for recent deforestation (default: 6)")
    
    # Testing Configuration
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--test_samples", type=int, default=50000, help="Max samples for testing phase to prevent OOM")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold locked from validation")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the resulting CSV")

    args = parser.parse_args()
    
    # Quick sanity check
    if args.lookback_months > args.context_months:
        raise ValueError(f"lookback_months ({args.lookback_months}) cannot be greater than context_months ({args.context_months})")
        
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

    in_channels = base_channels * seq_len if (args.mode == 'sequence' and args.model_type == 'resunet') else base_channels

    print(f"Detected Shape -> Base Channels: {base_channels}, In Channels: {in_channels}")

    # 4. Initialize Model
    model = get_model(args.model_type, in_channels, seq_len, device)
    print(f"Loading weights from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 5. Evaluate Emerging Frontiers
    print(f"Starting emerging frontiers evaluation (Threshold: {args.threshold})...")
    frontier_df = evaluate_new_frontiers(
        model, 
        test_loader, 
        device, 
        args.model_type, 
        args.mode, 
        args.threshold,
        args.lastmonth_channel,
        args.lookback_months
    )

    # 6. Output Results
    if not frontier_df.empty:
        print("\n=== Emerging Frontiers Performance ===")
        print(frontier_df.to_string(index=False))

        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, f"new_frontiers_results_{args.model_type}_{args.context_months}m_context_{args.lookback_months}m_lookback.csv")
        frontier_df.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()