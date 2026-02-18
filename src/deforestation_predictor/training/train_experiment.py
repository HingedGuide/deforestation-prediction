"""
Script Name: Deep Learning Training Pipeline for Deforestation Analysis

Description:
    This script trains deep learning models (2D ResUNet, 3D ResUNet, ConvLSTM, ViViT) 
    for deforestation prediction using satellite data. It supports both 2D (snapshot) 
    and 3D (spatiotemporal) training modes and handles multi-tile datasets.

    Key Features:
    - Multi-tile data loading with memory mapping (mmap) for efficiency.
    - Balanced sampling during training (50/50 forest/deforestation).
    - Unbalanced (representative) sampling during testing.
    - Support for 'snapshot' (2D) and 'sequence' (3D) modes.
    - Automatic handling of 'No Data' pixels in loss calculation.
    - Logs Precision, Recall, and Threshold alongside F0.5.
    - Early Stopping implemented to prevent overfitting.
"""

import argparse
import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_curve

# --- Project Imports ---
# Ensure these modules are accessible in your python path
from deforestation_predictor.models.architectures import ResUNet, ResUNet3D, ConvLSTM3D, ViViTSegmentation
from deforestation_predictor.training.loss import CombinedLoss 
from deforestation_predictor.utils.logger import setup_logger

# Fallback logger setup if not available in project structure
if 'setup_logger' not in locals():
    def setup_logger(name, log_file):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
        return logger

# Initialize logger
logger = setup_logger(__name__, log_file="training_experiment.log")


# ==============================================================================
# 1. Helper Classes (Dataset & EarlyStopping)
# ==============================================================================

class EarlyStopping:
    """
    Early stops the training if validation score doesn't improve after a given patience.
    """
    def __init__(self, patience=20, delta=0, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last time validation score improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.Inf

    def __call__(self, val_score):
        score = val_score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class MultiTileDataset(Dataset):
    """
    PyTorch Dataset for loading multi-temporal satellite data from multiple .npy tiles.
    """
    def __init__(self, img_paths, gt_paths, coordinates, size_crops=64, sequence_length=1, epoch_size=None):
        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.coords = coordinates
        self.size_crops = size_crops
        self.seq_len = sequence_length
        
        # Load files in read-only mmap mode to conserve RAM
        self.images = [np.load(p, mmap_mode='r') for p in img_paths]
        self.gts = [np.load(p, mmap_mode='r') for p in gt_paths]
        
        self.epoch_size = epoch_size if epoch_size else len(self.coords)

    def __len__(self):
        return min(len(self.coords), self.epoch_size)

    def __getitem__(self, idx):
        # Retrieve sample metadata
        tile_idx, t, x, y = self.coords[idx]
        x_end, y_end = x + self.size_crops, y + self.size_crops
        
        # --- Load Input Data (Variables) ---
        img_data = self.images[tile_idx][:, t : t + self.seq_len, x : x_end, y : y_end]
        X = torch.from_numpy(np.copy(img_data)).float()
        
        # If 2D mode (seq_len=1), remove the time dimension -> (Channels, H, W)
        if self.seq_len == 1:
            X = X.squeeze(1) 

        # --- Load Target Data (Ground Truth) ---
        target_t = t + self.seq_len - 1
        gt_data = self.gts[tile_idx][target_t, x : x_end, y : y_end]
        y = torch.from_numpy(np.copy(gt_data)).long()

        return X, y


# ==============================================================================
# 2. Coordinate Generator
# ==============================================================================

def get_coordinates(gt_paths, mask_paths, size_crops, sequence_length=1, balance=1, max_samples=None):
    all_coordinates = [] 
    logger.info(f"Scanning {len(gt_paths)} tiles (Seq Len: {sequence_length})...")

    if max_samples:
        logger.info(f"Subsampling enabled: Max {max_samples} total samples.")

    for tile_idx, (gt_path, mask_path) in enumerate(zip(gt_paths, mask_paths)):
        gt_array = np.load(gt_path, mmap_mode='r') 
        mask = np.load(mask_path, mmap_mode='r')
        mask_data = np.array(mask)
        
        mask_data[:size_crops, :] = 0; mask_data[-size_crops:, :] = 0
        mask_data[:, :size_crops] = 0; mask_data[:, -size_crops:] = 0

        max_time = gt_array.shape[0] - sequence_length + 1
        if max_time <= 0: continue

        for t in range(max_time):
            target_t = t + sequence_length - 1
            gt_slice = gt_array[target_t]
            
            if balance == 1:
                zeros = np.where((gt_slice == 0) & (mask_data == 1))
                ones = np.where((gt_slice == 1) & (mask_data == 1))
                
                zeros_coords = list(zip(zeros[0], zeros[1]))
                ones_coords = list(zip(ones[0], ones[1]))
                
                if len(zeros_coords) > 0 and len(ones_coords) > 0:
                    n_samples = min(len(zeros_coords), len(ones_coords))
                    zeros_coords = random.sample(zeros_coords, n_samples)
                    ones_coords = random.sample(ones_coords, n_samples)
                    
                    for r, c in zeros_coords: all_coordinates.append((tile_idx, t, r, c))
                    for r, c in ones_coords: all_coordinates.append((tile_idx, t, r, c))

            elif balance == 0:
                valid_mask = ((gt_slice == 0) | (gt_slice == 1)) & (mask_data == 1)
                rows, cols = np.where(valid_mask)
                total_valid = len(rows)
                
                if total_valid > 0:
                    if max_samples:
                        samples_to_take = min(total_valid, 2000) 
                        indices = np.random.choice(total_valid, samples_to_take, replace=False)
                        rows = rows[indices]
                        cols = cols[indices]
                    
                    for r, c in zip(rows, cols):
                        all_coordinates.append((tile_idx, t, r, c))

    random.shuffle(all_coordinates)
    
    if max_samples and len(all_coordinates) > max_samples:
        all_coordinates = all_coordinates[:max_samples]
        
    logger.info(f"Total samples collected: {len(all_coordinates)}")
    return all_coordinates


def fix_random_seeds(seed=31):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# ==============================================================================
# 3. Main Script
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deforestation Prediction Training")
    
    # Data Paths
    parser.add_argument('--image_path', type=str, default='./laura_preprocessing/output', help="Root folder for .npy files")
    parser.add_argument('--tiles', type=str, default="00N_000E", help="Comma-separated list of tile IDs")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    
    # Model Configuration
    parser.add_argument("--model_type", type=str, default="resunet", choices=["resunet", "resunet3d", "convlstm3d", "vivit"])
    parser.add_argument('--mode', type=str, default='snapshot', choices=['sequence', 'snapshot'], help="Training mode: 'snapshot' (2D) or 'sequence' (3D)")
    parser.add_argument("--context_months", type=int, default=12, help="Sequence length for 3D models")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples per training epoch")
    parser.add_argument("--balance", type=int, default=1, help="1 = Balanced training (50/50), 0 = Imbalanced")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    
    # Testing Configuration
    parser.add_argument("--test_samples", type=int, default=50000, help="Max samples for testing phase to prevent OOM")
    
    # Misc
    parser.add_argument("--seed", type=int, default=31, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    fix_random_seeds(args.seed)
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate File Paths
    tile_list = args.tiles.split(",")
    tr_img_paths = [os.path.join(args.image_path, f'{t}_var_train.npy') for t in tile_list]
    tr_gt_paths = [os.path.join(args.image_path, f'{t}_gt_tr.npy') for t in tile_list] 
    
    val_img_paths = [os.path.join(args.image_path, f'{t}_var_val.npy') for t in tile_list]
    val_gt_paths = [os.path.join(args.image_path, f'{t}_gt_val.npy') for t in tile_list]
    
    test_img_paths = [os.path.join(args.image_path, f'{t}_var_test.npy') for t in tile_list]
    test_gt_paths = [os.path.join(args.image_path, f'{t}_gt_test.npy') for t in tile_list]
    
    mask_paths = [os.path.join(args.image_path, f'{t}_mask.npy') for t in tile_list]

    # Determine Sequence Length
    seq_len = args.context_months if args.mode == 'sequence' else 1
    
    # --- Dataset Preparation ---
    logger.info("Generating Training Coordinates...")
    train_coords = get_coordinates(tr_gt_paths, mask_paths, 64, seq_len, args.balance)
    
    logger.info("Generating Validation Coordinates...")
    val_coords = get_coordinates(val_gt_paths, mask_paths, 64, seq_len, args.balance)

    train_ds = MultiTileDataset(tr_img_paths, tr_gt_paths, train_coords, 64, seq_len, args.samples)
    val_ds = MultiTileDataset(val_img_paths, val_gt_paths, val_coords, 64, seq_len, args.samples // 5)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- Model Setup ---
    sample_data = np.load(tr_img_paths[0], mmap_mode='r')
    base_channels = sample_data.shape[0]
    
    if args.mode == 'sequence' and args.model_type == 'resunet':
        in_channels = base_channels * seq_len
        raise TypeError("2D ResUNet in sequence mode requires explicit channel flattening.")
    else:
        in_channels = base_channels

    logger.info(f"Model: {args.model_type} | Mode: {args.mode} | In Channels: {in_channels}")

    if args.model_type == "resunet":
        model = ResUNet(in_channels=in_channels, num_classes=1)
    elif args.model_type == "resunet3d":
        model = ResUNet3D(in_channels=base_channels, time_depth=seq_len, num_classes=1)
    elif args.model_type == "convlstm3d":
        model = ConvLSTM3D(in_channels=base_channels, time_depth=seq_len, num_classes=1)
    elif args.model_type == "vivit":
        model = ViViTSegmentation(in_channels=base_channels, time_depth=seq_len, num_classes=1)

    model = model.to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    criterion = CombinedLoss().to(device) 

    # --- Training Loop ---
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    best_f05 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        random.shuffle(train_loader.dataset.coords)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            
            if args.model_type == 'resunet' and args.mode == 'sequence':
                b, c, t, h, w = X.shape
                X = X.view(b, c * t, h, w)
            
            optimizer.zero_grad()
            logits = model(X)
            
            if logits.shape[1] == 1: logits = logits.squeeze(1)
            
            valid_pixels = (y == 0) | (y == 1)
            weight_mask = torch.ones_like(y, dtype=torch.float)
            weight_mask[~valid_pixels] = 0.0
            
            y_safe = y.clone()
            y_safe[~valid_pixels] = 0
            
            loss = criterion(logits, y_safe, weight_mask)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation phase
        avg_val_loss, val_f05, val_prec, val_rec, val_thresh = validate(model, val_loader, criterion, device, args.model_type, args.mode)
        
        logger.info(
            f"Epoch {epoch+1}: Loss {train_loss/len(train_loader):.4f} | "
            f"Val F0.5: {val_f05:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | Thresh: {val_thresh:.4f}"
        )
        
        # Save best model
        if val_f05 > best_f05:
            best_f05 = val_f05
            save_path = os.path.join(args.save_dir, f"best_{args.model_type}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")

        # Early Stopping Check
        early_stopping(val_f05)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered! Model hasn't improved for {} epochs.".format(args.patience))
            break

    # --- Testing Phase ---
    logger.info(f"Starting Testing Phase (Max {args.test_samples} samples)...")
    
    test_coords = get_coordinates(test_gt_paths, mask_paths, 64, seq_len, balance=0, max_samples=args.test_samples)
    test_ds = MultiTileDataset(test_img_paths, test_gt_paths, test_coords, 64, seq_len, epoch_size=None)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    best_path = os.path.join(args.save_dir, f"best_{args.model_type}.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        logger.info(f"Loaded best model from {best_path}")
    else:
        logger.warning(f"Best model not found at {best_path}. Testing with last epoch weights.")
    
    # Calculate metrics on test set
    test_loss, test_f05, test_prec, test_rec, test_thresh = validate(model, test_loader, criterion, device, args.model_type, args.mode)
    
    logger.info("========================================")
    logger.info(f"FINAL TEST RESULTS ({args.model_type}):")
    logger.info(f"F0.5 Score: {test_f05:.4f}")
    logger.info(f"Precision:  {test_prec:.4f}")
    logger.info(f"Recall:     {test_rec:.4f}")
    logger.info(f"Threshold:  {test_thresh:.4f}")
    logger.info("========================================")


def validate(model, loader, criterion, device, model_type, mode):
    """
    Validation / Testing Loop.
    Calculates Loss and returns F0.5, Precision, Recall, and the Threshold used.
    """
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0
    
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Validating", leave=False):
            X, y = X.to(device), y.to(device)
            
            if model_type == 'resunet' and mode == 'sequence':
                b, c, t, h, w = X.shape
                X = X.view(b, c * t, h, w)
                
            logits = model(X)
            if logits.shape[1] == 1: logits = logits.squeeze(1)
            
            # --- Safe Loss Calculation ---
            valid_pixels = (y == 0) | (y == 1)
            weight_mask = torch.ones_like(y, dtype=torch.float)
            weight_mask[~valid_pixels] = 0.0
            y_safe = y.clone()
            y_safe[~valid_pixels] = 0
            
            loss = criterion(logits, y_safe, weight_mask)
            total_loss += loss.item()
            
            # Collect predictions for metric calculation
            probs = torch.sigmoid(logits)
            if valid_pixels.sum() > 0:
                all_preds.append(probs[valid_pixels].cpu().numpy())
                all_targets.append(y[valid_pixels].cpu().numpy())
                
    # Calculate Metrics
    f05 = 0.0
    precision_val = 0.0
    recall_val = 0.0
    best_threshold = 0.5
    
    if all_targets:
        y_true, y_scores = np.concatenate(all_targets), np.concatenate(all_preds)
        
        # Check if we have at least one positive and one negative sample to avoid sklearn errors
        if len(np.unique(y_true)) > 1:
            prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
            
            # F-beta score: (1 + beta^2) * (P * R) / ((beta^2 * P) + R)
            # Beta = 0.5 (Favors precision)
            fbeta = (1.25 * prec * rec) / (0.25 * prec + rec + 1e-8)
            
            # Find the index of the best F0.5 score
            ix = np.argmax(fbeta)
            
            f05 = fbeta[ix]
            precision_val = prec[ix]
            recall_val = rec[ix]
            
            # Thresholds array is always 1 shorter than prec/rec arrays in sklearn
            if ix < len(thresholds):
                best_threshold = thresholds[ix]
            else:
                best_threshold = 1.0 
            
    return total_loss/len(loader), f05, precision_val, recall_val, best_threshold


if __name__ == "__main__":
    main()