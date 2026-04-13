"""
Script Name: XGBoost Baseline Training Pipeline

Description:
    This script trains an XGBoost model as a baseline for deforestation prediction.
    It natively loads the processed multi-tile .npy arrays and uses the country-specific
    TIFF masks. The spatial and temporal dimensions are flattened to create 1D feature
    vectors per pixel, mimicking the DL sequence context length.
    
    Key Features:
    - Native support for .npy variable and ground truth arrays.
    - Rasterio integration for precise country masking.
    - Automatic temporal slicing to match DL context lengths.
    - Validation-locked threshold to prevent data leakage during testing.
"""

import argparse
import os
import gc
import sys
import logging
import rasterio
import joblib
import numpy as np
import xgboost as xgb
import wandb
from pathlib import Path
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

# ------------- CONFIG ------------- #
# Cluster Settings
N_ESTIMATORS = 200
MAX_DEPTH = 8
VAL_TEST_SAMPLE_RATE = 0.2  # Percentage of random valid pixels sampled for val/test
TRAIN_SAMPLE_RATE = 0.05    # Downsampling rate for negative (forest) pixels during training


# ------------- LOGGING SETUP ------------- #
def setup_logger(log_file):
    """
    Configures the logger to output to both the console and a specific file.
    """
    logger = logging.getLogger("xgboost_baseline")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicate logs when running multiple times
    if logger.handlers:
        logger.handlers.clear()

    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file)
    
    log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)
    
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger


# ------------- HELPER: FOCAL LOSS ------------- #
def calculate_focal_loss(y_true, y_pred_prob, alpha=0.25, gamma=2.0):
    """
    Calculates the focal loss for binary classification.
    """
    p = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)
    ce_loss = - (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    pt = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    focal_loss = alpha_t * (1 - pt) ** gamma * ce_loss
    return np.mean(focal_loss)


# ------------- DATA LOADING ------------- #
def load_and_flatten_data(image_path, tile_list, country, split_name, context_length=12, sample_rate=0.1, balanced=True):
    """
    Loads spatial data from .npy files, applies temporal slicing, uses the TIFF mask,
    and flattens the result into 2D tabular data [Samples, Features] for XGBoost.
    """
    logger = logging.getLogger("xgboost_baseline")
    logger.info(f"Loading {split_name} data (Context: {context_length}m)...")

    X_list = []
    y_list = []

    for tile_idx, t_id in enumerate(tile_list):
        img_file = os.path.join(image_path, f'{t_id}_var_{split_name}.npy')
        gt_file = os.path.join(image_path, f'{t_id}_gt_{split_name}.npy')
        mask_file = os.path.join(image_path, f'{t_id}_mask_{country}.tiff')

        if not (os.path.exists(img_file) and os.path.exists(gt_file) and os.path.exists(mask_file)):
            logger.warning(f"Missing files for tile {t_id}. Skipping...")
            continue

        # Load arrays using memory mapping to conserve RAM
        var_array = np.load(img_file, mmap_mode='r')
        gt_array = np.load(gt_file, mmap_mode='r')
        
        with rasterio.open(mask_file) as src:
            mask_data = src.read(1)

        # Calculate max starting time to fit the sequence length
        max_time = gt_array.shape[0] - context_length + 1
        if max_time <= 0:
            continue

        # Iterate over all possible temporal windows
        for t in range(max_time):
            target_t = t + context_length - 1
            y_slice = gt_array[target_t]

            # Valid pixels are defined by actual ground truth and the country mask
            valid_pixels = ((y_slice == 0) | (y_slice == 1)) & (mask_data == 1)
            
            if balanced:
                # Downsample majority class (0) and keep all minority class (1)
                zeros = np.where((y_slice == 0) & valid_pixels)
                ones = np.where((y_slice == 1) & valid_pixels)
                
                n_pos = len(ones[0])
                n_neg = len(zeros[0])
                
                if n_pos > 0 and n_neg > 0:
                    n_neg_sample = int(n_neg * sample_rate)
                    if n_neg_sample > 0:
                        neg_sample_idx = np.random.choice(n_neg, size=n_neg_sample, replace=False)
                        sampled_zeros_r = zeros[0][neg_sample_idx]
                        sampled_zeros_c = zeros[1][neg_sample_idx]
                        
                        rows = np.concatenate([ones[0], sampled_zeros_r])
                        cols = np.concatenate([ones[1], sampled_zeros_c])
                    else:
                        rows, cols = ones[0], ones[1]
                else:
                    continue
            else:
                # Random sampling of all valid pixels for val/test
                valid_idx = np.where(valid_pixels)
                n_valid = len(valid_idx[0])
                
                if n_valid > 0:
                    n_keep = int(n_valid * sample_rate)
                    if n_keep > 0:
                        keep_idx = np.random.choice(n_valid, size=n_keep, replace=False)
                        rows = valid_idx[0][keep_idx]
                        cols = valid_idx[1][keep_idx]
                    else:
                        continue
                else:
                    continue

            # Extract features for the sampled pixels
            # Shape extracted: [Channels, context_length, Num_Samples]
            X_samples = var_array[:, t : target_t + 1, rows, cols]
            
            # Transpose to [Num_Samples, Channels, context_length] and flatten features
            X_samples = X_samples.transpose(2, 0, 1).reshape(len(rows), -1)
            y_samples = y_slice[rows, cols]

            X_list.append(X_samples)
            y_list.append(y_samples)

    if not X_list:
        return None, None
        
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ------------- MAIN EXECUTION ------------- #
def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Baseline")
    
    # Data Paths
    parser.add_argument('--image_path', type=str, default='./laura_preprocessing/output', help="Root folder for .npy files")
    parser.add_argument('--tiles', type=str, default="00N_000E", help="Comma-separated list of tile IDs")
    parser.add_argument('--country', type=str, default="Gabon", help="Country name used in the TIFF mask filename")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--context_months", type=int, default=12, help="Number of past months to use (3, 6, 12)")
    
    args = parser.parse_args()

    # Setup Dynamic Logging
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    log_file_path = log_dir / f"xgboost_{args.context_months}m.log"
    logger = setup_logger(str(log_file_path))

    run_name = f"XGBoost_{args.context_months}m"
    logger.info(f"Starting {run_name} using data from {args.image_path}")

    # Initialize W&B
    wandb.init(
        project="deforestation-prediction", 
        name=run_name, 
        tags=["baseline", "xgboost", "optimized", args.country],
        config=vars(args)
    )

    tile_list = args.tiles.split(",")

    # 1. Load Training Data
    X_train, y_train = load_and_flatten_data(
        image_path=args.image_path,
        tile_list=tile_list,
        country=args.country,
        split_name="train", 
        context_length=args.context_months, 
        sample_rate=TRAIN_SAMPLE_RATE,
        balanced=True
    )

    # 2. Load Validation Data
    X_val, y_val = load_and_flatten_data(
        image_path=args.image_path,
        tile_list=tile_list,
        country=args.country,
        split_name="val", 
        context_length=args.context_months, 
        sample_rate=VAL_TEST_SAMPLE_RATE, 
        balanced=False
    )

    if X_train is not None and X_val is not None:
        logger.info(f"Training XGBoost ({X_train.shape[0]} samples)...")
        logger.info(f"Feature count per pixel: {X_train.shape[1]}")

        # Class imbalance correction
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        logger.info(f"Class Balance - Pos: {n_pos}, Neg: {n_neg}")
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            tree_method='hist',
            device='cpu',
            eval_metric=["logloss", "aucpr"]
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        # Log History to W&B
        results = model.evals_result()
        epochs = len(results['validation_0']['logloss'])
        for i in range(epochs):
            wandb.log({
                "train_logloss": results['validation_0']['logloss'][i],
                "val_logloss": results['validation_1']['logloss'][i],
                "train_aucpr": results['validation_0']['aucpr'][i],
                "val_aucpr": results['validation_1']['aucpr'][i],
            }, step=i)

        # Save model
        model_filename = os.path.join(args.save_dir, f"xgboost_{args.context_months}m.pkl")
        joblib.dump(model, model_filename)
        logger.info(f"Model saved as {model_filename}")

        # --- VALIDATION: Calculate Optimal Threshold ---
        logger.info("Calculating optimal threshold on Validation Data...")
        probs_val = model.predict_proba(X_val)[:, 1]
        precisions_val, recalls_val, thresholds_val = precision_recall_curve(y_val, probs_val)
        
        beta = 0.5
        fbeta_scores_val = (1 + beta ** 2) * (precisions_val * recalls_val) / ((beta ** 2 * precisions_val) + recalls_val + 1e-8)
        best_idx_val = np.argmax(fbeta_scores_val)
        locked_threshold = thresholds_val[best_idx_val]
        
        logger.info(f"Locked Threshold from Validation: {locked_threshold:.4f}")

        # Cleanup memory before testing
        del X_train, y_train, X_val, y_val
        gc.collect()

        # 3. Final Evaluation on TEST SET
        logger.info("Loading TEST Data...")
        X_test, y_test = load_and_flatten_data(
            image_path=args.image_path,
            tile_list=tile_list,
            country=args.country,
            split_name="test", 
            context_length=args.context_months, 
            sample_rate=VAL_TEST_SAMPLE_RATE, 
            balanced=False
        )

        if X_test is not None:
            logger.info("Calculating Final Test Metrics...")
            probs_test = model.predict_proba(X_test)[:, 1]

            # Apply locked threshold to test set predictions
            preds_test = (probs_test >= locked_threshold).astype(int)
            
            test_precision = precision_score(y_test, preds_test, zero_division=0)
            test_recall = recall_score(y_test, preds_test, zero_division=0)
            test_f05 = fbeta_score(y_test, preds_test, beta=0.5, zero_division=0)
            test_focal_loss = calculate_focal_loss(y_test, probs_test)

            logger.info("========================================")
            logger.info("FINAL TEST RESULTS (XGBoost):")
            logger.info(f"F0.5 Score: {test_f05:.4f}")
            logger.info(f"Precision:  {test_precision:.4f}")
            logger.info(f"Recall:     {test_recall:.4f}")
            logger.info(f"Threshold:  {locked_threshold:.4f} (Locked from Validation)")
            logger.info("========================================")
            
            wandb.log({
                "test_focal_loss": test_focal_loss,
                "locked_threshold": locked_threshold,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f05": test_f05,
                "test_pr_curve": wandb.plot.pr_curve(
                    y_test,
                    np.stack([1 - probs_test, probs_test], axis=1),
                    labels=["Forest", "Deforestation"]
                )
            })

    wandb.finish()

if __name__ == "__main__":
    main()