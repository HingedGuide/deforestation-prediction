import numpy as np
import xgboost as xgb
import wandb
from pathlib import Path
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import joblib
import logging
import sys
import gc
import argparse

# ------------- CONFIG ------------- #
# Calculate absolute project root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Default data root
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed_3d" / "GABON"

IGNORE_LABEL = 2

# Cluster Settings
N_ESTIMATORS = 200
MAX_DEPTH = 8  # Increased from 6 to 8 to capture subtler 3m patterns
VAL_TEST_SAMPLE_RATE = 0.2
TRAIN_SAMPLE_RATE = 0.05 # Decreased from 0.1 to 0.05 to reduce class imbalance


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
    
    log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
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
def load_and_flatten_data(data_root, split_name, context_length=12, sample_rate=0.1, balanced=True):
    """
    Loads 3D patch data, applies temporal slicing (RQ2), and flattens to 2D for XGBoost.
    """
    logger = logging.getLogger("xgboost_baseline")
    split_dir = Path(data_root) / split_name

    if not split_dir.exists():
        logger.error(f"Directory not found: {split_dir}")
        return None, None

    files = list(split_dir.glob("*.npz"))
    logger.info(f"Loading {split_name} data from {len(files)} patches (Context: {context_length}m)...")

    X_list = []
    y_list = []

    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                X_cube = data['X']  # [Channels, Time, Height, Width]
                y_mask = data['y']

                # --- RQ2: Temporal Window Slicing ---
                # Take only the LAST 'context_length' months
                current_T = X_cube.shape[1]
                if current_T >= context_length:
                    X_cube = X_cube[:, -context_length:, :, :]
                
                V, T, H, W = X_cube.shape

                # Flatten [V, T, H, W] -> [Pixels, Features]
                # Features = V * T (e.g., 10 variables * 3 months = 30 features per pixel)
                X_flat = X_cube.transpose(2, 3, 0, 1).reshape(H * W, -1)
                y_flat = y_mask.reshape(-1)

                valid_mask = y_flat != IGNORE_LABEL
                X_flat = X_flat[valid_mask]
                y_flat = y_flat[valid_mask]

                if len(y_flat) == 0: continue

                # --- SAMPLING LOGIC ---
                rng = np.random.default_rng()

                if balanced and split_name == 'train':
                    # TRAINING: Downsample negatives to balance classes (keep all positives)
                    pos_indices = np.where(y_flat == 1)[0]
                    neg_indices = np.where(y_flat == 0)[0]
                    n_neg = int(len(neg_indices) * sample_rate)

                    if n_neg > 0:
                        neg_sample = rng.choice(neg_indices, size=n_neg, replace=False)
                        indices = np.concatenate([pos_indices, neg_sample])
                        rng.shuffle(indices)
                        X_flat = X_flat[indices]
                        y_flat = y_flat[indices]

                elif sample_rate < 1.0:
                    # VAL/TEST: Randomly keep X% of ALL pixels to save memory
                    n_keep = int(len(y_flat) * sample_rate)
                    if n_keep > 0:
                        indices = rng.choice(len(y_flat), size=n_keep, replace=False)
                        X_flat = X_flat[indices]
                        y_flat = y_flat[indices]

                X_list.append(X_flat)
                y_list.append(y_flat)
        except Exception as e:
            logger.warning(f"Error loading {f.name}: {e}")
            pass

    if not X_list: return None, None
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ------------- MAIN EXECUTION ------------- #
if __name__ == "__main__":
    
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Train XGBoost Baseline")
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT), help="Path to processed data")
    parser.add_argument("--context_months", type=int, default=12, help="Number of past months to use (3, 6, 12)")
    args = parser.parse_args()

    # 2. Setup Dynamic Logging
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set dynamic log file name
    log_file_path = log_dir / f"xgboost_{args.context_months}m.log"
    logger = setup_logger(str(log_file_path))

    run_name = f"XGBoost_{args.context_months}m"
    logger.info(f"Starting {run_name} using data from {args.data_root}")

    # 3. Initialize W&B
    wandb.init(
        project="deforestation-prediction", 
        name=run_name, 
        tags=["baseline", "xgboost", "optimized"],
        config=vars(args)
    )

    # 4. Load Training Data
    logger.info("Step 1: Preparing Training Data...")
    X_train, y_train = load_and_flatten_data(
        args.data_root, 
        "train", 
        context_length=args.context_months, 
        sample_rate=TRAIN_SAMPLE_RATE,  # Use lower sample rate to reduce imbalance
        balanced=True
    )

    # 5. Load Validation Data
    logger.info("Step 1b: Preparing Validation Data...")
    X_val, y_val = load_and_flatten_data(
        args.data_root, 
        "val", 
        context_length=args.context_months, 
        sample_rate=VAL_TEST_SAMPLE_RATE, 
        balanced=False
    )

    if X_train is not None and X_val is not None:
        logger.info(f"Step 2: Training XGBoost on GPU ({X_train.shape[0]} samples)...")
        logger.info(f"Feature count per pixel: {X_train.shape[1]}")

        # --- CLASS IMBALANCE CORRECTION ---
        # Calculate scale_pos_weight dynamically based on the loaded training data
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        logger.info(f"Class Balance - Pos: {n_pos}, Neg: {n_neg}")
        logger.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, # Apply weight to prioritize minority class (deforestation)
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

        # Save model with context length in name
        model_filename = f"xgboost_{args.context_months}m.pkl"
        joblib.dump(model, model_filename)
        logger.info(f"Model saved as {model_filename}. Cleaning memory...")

        # --- MEMORY CLEANUP ---
        del X_train, y_train, X_val, y_val
        gc.collect()

        # 6. Final Evaluation on TEST SET
        logger.info("Step 4: Loading TEST Data...")
        X_test, y_test = load_and_flatten_data(
            args.data_root, 
            "test", 
            context_length=args.context_months, 
            sample_rate=VAL_TEST_SAMPLE_RATE, 
            balanced=False
        )

        if X_test is not None:
            logger.info("Calculating Final Test Metrics...")
            probs_test = model.predict_proba(X_test)[:, 1]

            # Metrics
            test_focal_loss = calculate_focal_loss(y_test, probs_test)
            precisions, recalls, thresholds = precision_recall_curve(y_test, probs_test)

            beta = 0.5
            fbeta_scores = (1 + beta ** 2) * (precisions * recalls) / ((beta ** 2 * precisions) + recalls + 1e-8)
            best_idx = np.argmax(fbeta_scores)
            best_threshold = thresholds[best_idx]

            preds_test = (probs_test >= best_threshold).astype(int)
            test_precision = precision_score(y_test, preds_test)
            test_recall = recall_score(y_test, preds_test)
            test_f05 = fbeta_scores[best_idx]

            logger.info(f"Final Test Threshold: {best_threshold:.4f}")
            logger.info(f"Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f} | Test F0.5: {test_f05:.4f}")
            
            wandb.log({
                "test_focal_loss": test_focal_loss,
                "best_threshold": best_threshold,
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