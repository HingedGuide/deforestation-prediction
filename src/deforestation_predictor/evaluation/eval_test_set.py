"""
Script: Evaluate Test Set Metrics (Legacy / MultiTile Mode)
Description:
    Evaluates trained models using the MultiTileDataset logic compatible with 
    'train_experiment.py' and the 'laura_preprocessing' data format.
"""

import argparse
import torch
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, fbeta_score, average_precision_score

# Project imports
from deforestation_predictor.models.architectures import (
    ResUNet, ResUNet3D, ViViTSegmentation, ConvLSTM3D
)
from deforestation_predictor.training.train_experiment import MultiTileDataset, get_coordinates


# ==============================================================================
# 2. Main Evaluation Logic
# ==============================================================================

def get_model(model_type, in_channels, time_depth, num_classes=1):
    if model_type == "resunet":
        return ResUNet(in_channels=in_channels, num_classes=num_classes)
    elif model_type == "resunet3d":
        return ResUNet3D(in_channels=in_channels, time_depth=time_depth, num_classes=num_classes)
    elif model_type == "convlstm3d":
        return ConvLSTM3D(in_channels=in_channels, time_depth=time_depth, num_classes=num_classes)
    elif model_type == "vivit":
        return ViViTSegmentation(in_channels=in_channels, time_depth=time_depth, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate(model, loader, device, threshold, model_type, mode):
    model.eval()
    all_probs = []
    all_targets = []
    
    print(f"Starting evaluation with threshold: {threshold}...")
    
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating"):
            X = X.to(device)
            y = y.to(device)

            # Flatten logic for ResUNet Sequence mode
            if model_type == 'resunet' and mode == 'sequence':
                if X.ndim == 5: # [B, C, T, H, W]
                    b, c, t, h, w = X.shape
                    X = X.view(b, c * t, h, w)
            
            logits = model(X)
            if logits.shape[1] == 1: logits = logits.squeeze(1)
            
            probs = torch.sigmoid(logits)
            
            # Valid mask (0 or 1 are valid, anything else is ignore)
            valid_mask = (y == 0) | (y == 1)
            
            if valid_mask.sum() == 0: continue
                
            valid_probs = probs[valid_mask]
            valid_targets = y[valid_mask]
            
            all_probs.append(valid_probs.cpu().numpy())
            all_targets.append(valid_targets.cpu().numpy())

    if not all_targets:
        print("No valid targets found.")
        return None

    y_true = np.concatenate(all_targets)
    y_scores = np.concatenate(all_probs)
    
    print(f"Total valid pixels evaluated: {len(y_true)}")
    
    y_pred = (y_scores >= threshold).astype(int)
    
    return {
        "F0.5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "PR-AUC": average_precision_score(y_true, y_scores)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate (Legacy/MultiTile)")
    
    # Path args match train_experiment.py
    parser.add_argument('--image_path', type=str, required=True, help="Folder with .npy files (e.g. laura_preprocessing_3d/output_3d)")
    parser.add_argument('--tiles', type=str, default="00N_000E", help="Comma-separated tile IDs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth file")
    
    # Config
    parser.add_argument("--model_type", type=str, required=True, choices=["resunet", "resunet3d", "convlstm3d", "vivit"])
    parser.add_argument('--mode', type=str, default='snapshot', choices=['sequence', 'snapshot'])
    parser.add_argument("--context_months", type=int, default=12)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_samples", type=int, default=20000, help="Max number of patch samples")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Data Paths (Same logic as train_experiment)
    tile_list = args.tiles.split(",")
    test_img_paths = [os.path.join(args.image_path, f'{t}_var_test.npy') for t in tile_list]
    test_gt_paths = [os.path.join(args.image_path, f'{t}_gt_test.npy') for t in tile_list]
    mask_paths = [os.path.join(args.image_path, f'{t}_mask.npy') for t in tile_list]

    # Validate paths
    if not os.path.exists(test_img_paths[0]):
        raise FileNotFoundError(f"Could not find {test_img_paths[0]}. Check --image_path.")

    seq_len = args.context_months if args.mode == 'sequence' else 1
    
    # 2. Setup Dataset
    # balance=0 for testing (unbalanced/representative)
    coords = get_coordinates(test_gt_paths, mask_paths, 64, seq_len, balance=0, max_samples=args.test_samples)
    
    test_ds = MultiTileDataset(test_img_paths, test_gt_paths, coords, 64, seq_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. Setup Model
    # Load first sample to check channels
    sample_data = np.load(test_img_paths[0], mmap_mode='r')
    base_channels = sample_data.shape[0]
    
    if args.model_type == 'resunet' and args.mode == 'sequence':
        # Special case: 2D model with flattened time
        model_in_channels = base_channels * seq_len
    else:
        model_in_channels = base_channels
        
    print(f"Initializing {args.model_type} (In: {model_in_channels}, Depth: {seq_len})...")
    model = get_model(args.model_type, model_in_channels, seq_len)
    
    # Load Weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if list(checkpoint.keys())[0].startswith('module.'):
        checkpoint = {k[7:]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.to(device)
    
    # 4. Run
    metrics = evaluate(model, test_loader, device, args.threshold, args.model_type, args.mode)
    
    if metrics:
        print("\n" + "="*30)
        print(f"RESULTS ({args.model_type})")
        print("="*30)
        print(f"F0.5 Score:    {metrics['F0.5']:.4f}")
        print(f"Precision:     {metrics['Precision']:.4f}")
        print(f"Recall:        {metrics['Recall']:.4f}")
        print(f"PR-AUC:        {metrics['PR-AUC']:.4f}")
        print("="*30 + "\n")

if __name__ == "__main__":
    main()