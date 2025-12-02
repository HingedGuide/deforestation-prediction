import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, fbeta_score
import joblib
import logging
import sys

# ------------- CONFIG ------------- #
DATA_ROOT = Path("data/processed/GABON/3d_dataset")  # Adjust to your region
IGNORE_LABEL = 2
LOG_FILE = "xgboost_baseline.log"


# ------------- LOGGING SETUP ------------- #
def setup_logger(log_file):
    """
    Sets up a logger that writes to both the console (stdout) and a file.
    """
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times if the script is run repeatedly in some environments
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_file)

        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


logger = setup_logger(LOG_FILE)


# ------------- DATA LOADING ------------- #
def load_and_flatten_data(split_name, sample_rate=0.1, balanced=True):
    """
    Loads .npz patches and converts them to a tabular format for XGBoost.

    Args:
        split_name: 'train', 'val', or 'test'
        sample_rate: Fraction of negative pixels to keep (downsampling).
        balanced: If True, keeps all positives and downsamples negatives.
    """
    split_dir = DATA_ROOT / split_name
    files = list(split_dir.glob("*.npz"))

    X_list = []
    y_list = []

    logger.info(f"Loading {split_name} data from {len(files)} patches...")

    for i, f in enumerate(files):
        try:
            with np.load(f, allow_pickle=True) as data:
                # Shape: (V, T, H, W)
                X_cube = data['X']
                # Shape: (H, W)
                y_mask = data['y']

                # Dimensions
                V, T, H, W = X_cube.shape

                # Reshape features: We want one row per pixel.
                # Transpose to (H, W, V, T) -> reshape to (N_pixels, N_features)
                # Features = all variables across all timesteps flattened.
                X_flat = X_cube.transpose(2, 3, 0, 1).reshape(H * W, -1)
                y_flat = y_mask.reshape(-1)

                # Filter Ignore Labels
                valid_mask = y_flat != IGNORE_LABEL
                X_flat = X_flat[valid_mask]
                y_flat = y_flat[valid_mask]

                if len(y_flat) == 0:
                    continue

                # Subsampling logic to handle class imbalance/memory
                if balanced and split_name == 'train':
                    pos_indices = np.where(y_flat == 1)[0]
                    neg_indices = np.where(y_flat == 0)[0]

                    # Keep all positives
                    # Randomly sample negatives
                    n_neg = int(len(neg_indices) * sample_rate)

                    if n_neg > 0:
                        rng = np.random.default_rng()
                        neg_sample = rng.choice(neg_indices, size=n_neg, replace=False)
                        indices = np.concatenate([pos_indices, neg_sample])
                        rng.shuffle(indices)

                        X_flat = X_flat[indices]
                        y_flat = y_flat[indices]

                X_list.append(X_flat)
                y_list.append(y_flat)

        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

        # Periodic logging for long running processes
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(files)} files for {split_name}...")

    if not X_list:
        logger.error(f"No valid data found for {split_name}.")
        return None, None

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    return X_all, y_all


# ------------- MAIN EXECUTION ------------- #
if __name__ == "__main__":
    # 1. Prepare Training Data
    logger.info("Step 1: Preparing Training Data...")

    # NOTE: Adjust sample_rate based on your RAM. 0.1 means keep 10% of negative pixels.
    X_train, y_train = load_and_flatten_data("train", sample_rate=0.1, balanced=True)

    if X_train is not None:
        logger.info(f"Training Data Shape: {X_train.shape}")
        logger.info(f"Class balance: {np.mean(y_train):.2%} positive pixels")

        # 2. Train XGBoost
        logger.info("Step 2: Training XGBoost model...")

        # Hyperparameters can be tuned. 'hist' tree method is much faster for large data.
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            tree_method='hist'
        )
        model.fit(X_train, y_train)

        # Save model
        model_path = "xgboost_baseline.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # 3. Optimize Threshold on Validation
        logger.info("Step 3: Optimizing Threshold on Validation Data...")
        # For validation, we typically want the real distribution (balanced=False, sample_rate=1.0)
        # However, if memory is tight, you might need to sample val too.
        X_val, y_val = load_and_flatten_data("val", sample_rate=1.0, balanced=False)

        if X_val is not None:
            # Predict probabilities
            probs_val = model.predict_proba(X_val)[:, 1]

            # Calculate F0.5 for all thresholds
            precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)

            # F0.5 Formula: (1 + 0.5^2) * (P * R) / ((0.5^2 * P) + R)
            beta = 0.5
            fbeta_scores = (1 + beta ** 2) * (precisions * recalls) / ((beta ** 2 * precisions) + recalls + 1e-8)

            # Locate best threshold
            best_idx = np.argmax(fbeta_scores)
            best_threshold = thresholds[best_idx]
            best_f05 = fbeta_scores[best_idx]

            logger.info(f"Optimal Threshold found: {best_threshold:.4f}")
            logger.info(f"Max Validation F0.5 Score: {best_f05:.4f}")

            # 4. Evaluate on Test
            logger.info("Step 4: Evaluating on Test Data...")
            X_test, y_test = load_and_flatten_data("test", sample_rate=1.0, balanced=False)

            if X_test is not None:
                probs_