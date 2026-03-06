"""
Script Name: Patch Size Evaluation Analysis

Description:
    This standalone script evaluates a trained deep learning model on the test
    dataset and calculates the Recall broken down by the size of the 
    deforestation patches (connected components) in the ground truth.
    
    Patch size categories:
    - Small: 1 to 9 pixels
    - Medium: 10 to 49 pixels
    - Large: >= 50 pixels
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import recall_score
from scipy.ndimage import label

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

def evaluate_patch_size(model, loader, device, model_type, mode, threshold):
    """
    Evaluates the model on the test set and calculates Recall per patch size category.
    Uses scipy.ndimage.label to find connected components of deforestation.
    """
    model.eval()
    
    # Define the distance categories based on pixel count thresholds
    cat_dict = {
        'Small (< 10 px)': {'preds': [], 'targets': []},
        'Medium (10 - 49 px)': {'preds': [], 'targets': []},
        'Large (>= 50 px)': {'preds': [], 'targets': []}
    }

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Testing Patch Sizes"):
            X, y = X.to(device), y.to(device)

            # Handle temporal sequence dimensions
            if mode == 'sequence':
                b, c, t, h, w = X.shape
                X_input = X.view(b, c * t, h, w) if model_type == 'resunet' else X
            else:
                X_input = X

            # Forward pass
            logits = model(X_input)
            if logits.shape[1] == 1:
                logits = logits.squeeze(1)

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = y.cpu().numpy()

            for i in range(X.shape[0]):
                target_mask = targets[i]
                prob_mask = probs[i]
                
                # We only isolate the actual deforestation pixels (y == 1)
                def_pixels = (target_mask == 1)
                
                # If there is no deforestation in this patch, skip
                if not def_pixels.any():
                    continue
                
                # Label connected components (clusters of 1s)
                # Structure matrix defines connectivity (default is 4-way, can be changed to 8-way)
                labeled_array, num_features = label(def_pixels)
                
                # Iterate over each unique deforestation patch found
                for patch_idx in range(1, num_features + 1):
                    patch_locs = (labeled_array == patch_idx)
                    patch_size = patch_locs.sum()
                    
                    # Categorize based on pixel count
                    if patch_size < 10:
                        cat = 'Small (< 10 px)'
                    elif patch_size < 50:
                        cat = 'Medium (10 - 49 px)'
                    else:
                        cat = 'Large (>= 50 px)'
                        
                    # Store the predictions and actuals for this specific patch
                    cat_dict[cat]['preds'].append(prob_mask[patch_locs])
                    cat_dict[cat]['targets'].append(target_mask[patch_locs])

    # Calculate metrics (Mainly Recall, as Precision/F0.5 require tracking False Positives 
    # which do not belong to a specific ground truth patch size)
    results = []
    for cat, data in cat_dict.items():
        if not data['targets']:
            continue

        y_true = np.concatenate(data['targets'])
        y_scores = np.concatenate(data['preds'])
        y_pred = (y_scores >= threshold).astype(int)

        rec = recall_score(y_true, y_pred, zero_division=0)

        results.append({
            'Patch Size': cat,
            'Recall': rec,
            'Total_Deforested_Pixels': len(y_true)
        })

    df = pd.DataFrame(results)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate model recall per deforestation patch size.")
    
    # Data Paths
    parser.add_argument('--image_path', type=str, default='./laura_preprocessing/output', help="Root folder for .npy files")
    parser.add_argument('--tiles', type=str, default="00N_000E", help="Comma-separated list of tile IDs")
    parser.add_argument('--country', type=str, default="Laos", help="Country name used in the TIFF mask filename")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    
    # Model Configuration
    parser.add_argument("--model_type", type=str, required=True, choices=["resunet", "resunet3d", "convlstm3d", "vivit"])
    parser.add_argument("--mode", type=str, default="sequence", choices=["sequence", "snapshot"])
    parser.add_argument("--context_months", type=int, default=12, help="Sequence length for 3D models")
    
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

    in_channels = base_channels * seq_len if (args.mode == 'sequence' and args.model_type == 'resunet') else base_channels

    print(f"Detected Shape -> Base Channels: {base_channels}, In Channels: {in_channels}")

    # 4. Initialize Model
    model = get_model(args.model_type, in_channels, seq_len, device)
    print(f"Loading weights from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # 5. Evaluate Patch Sizes
    print(f"Starting patch size evaluation (Threshold: {args.threshold})...")
    patch_df = evaluate_patch_size(model, test_loader, device, args.model_type, args.mode, args.threshold)

    # 6. Output Results
    print("\n=== Patch Size Performance ===")
    print(patch_df.to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"patch_size_results_{args.model_type}_{args.context_months}m.csv")
    patch_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

if __name__ == "__main__":
    main()