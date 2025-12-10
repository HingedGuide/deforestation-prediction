import numpy as np
import xgboost as xgb
import wandb
from wandb.integration.xgboost import WandbCallback
from pathlib import Path
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, fbeta_score
import joblib
import logging
import sys

# ------------- CONFIG ------------- #
DATA_ROOT = Path("data/processed/GABON/3d_dataset")
IGNORE_LABEL = 2
LOG_FILE = "xgboost_baseline.log"


# ------------- LOGGING SETUP ------------- #
def setup_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_file)
        log_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    return logger


logger = setup_logger(LOG_FILE)


# ------------- HELPER: FOCAL LOSS ------------- #
def calculate_focal_loss(y_true, y_pred_prob, alpha=0.25, gamma=2.0):
    """
    Manually calculates Focal Loss for XGBoost predictions so we can compare
    it with the Deep Learning models.
    """
    # Clip probabilities to avoid log(0)
    p = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)

    # Calculate Cross Entropy terms
    ce_loss = - (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    # Calculate weights (pt)
    pt = np.where(y_true == 1, p, 1 - p)

    # Focal Loss formula: alpha * (1-pt)^gamma * CE
    # Note: simple alpha version (alpha for class 1, 1-alpha for class 0)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    focal_loss = alpha_t * (1 - pt) ** gamma * ce_loss

    return np.mean(focal_loss)


# ------------- DATA LOADING ------------- #
def load_and_flatten_data(split_name, sample_rate=0.1, balanced=True):
    # [Use the exact same loading code from your previous file]
    # ... (Code omitted for brevity, copy the load_and_flatten_data function from previous saved file) ...
    # RE-INSERT FULL FUNCTION HERE IF YOU NEED ME TO WRITE IT OUT AGAIN
    split_dir = DATA_ROOT / split_name
    files = list(split_dir.glob("*.npz"))
    X_list = []
    y_list = []

    logger.info(f"Loading {split_name} data from {len(files)} patches...")

    for i, f in enumerate(files):
        try:
            with np.load(f, allow_pickle=True) as data:
                X_cube = data['X']
                y_mask = data['y']
                V, T, H, W = X_cube.shape
                X_flat = X_cube.transpose(2, 3, 0, 1).reshape(H * W, -1)
                y_flat = y_mask.reshape(-1)

                valid_mask = y_flat != IGNORE_LABEL
                X_flat = X_flat[valid_mask]
                y_flat = y_flat[valid_mask]

                if len(y_flat) == 0: continue

                if balanced and split_name == 'train':
                    pos_indices = np.where(y_flat == 1)[0]
                    neg_indices = np.where(y_flat == 0)[0]
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
            pass  # Skip corrupt files

    if not X_list: return None, None
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ------------- MAIN EXECUTION ------------- #
if __name__ == "__main__":

    # 1. Initialize W&B
    wandb.init(project="deforestation-prediction", name="xgboost_baseline", tags=["baseline"])

    # 2. Prepare Data
    logger.info("Step 1: Preparing Training Data...")
    X_train, y_train = load_and_flatten_data("train", sample_rate=0.1, balanced=True)

    # Load Validation Data (Needed for live metrics)
    logger.info("Step 1b: Preparing Validation Data...")
    X_val, y_val = load_and_flatten_data("val", sample_rate=1.0, balanced=False)

    if X_train is not None and X_val is not None:
        logger.info("Step 2: Training XGBoost model...")

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            tree_method='hist',
            # METRIC UPDATE: Log 'aucpr' (PR-AUC) and 'logloss'
            eval_metric=["logloss", "aucpr"]
        )

        # Train with W&B Callback
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            callbacks=[WandbCallback()]
        )

        # Save model
        joblib.dump(model, "xgboost_baseline.pkl")

        # 3. Calculate Custom Metrics (Focal Loss, Precision, Recall)
        logger.info("Step 3: Calculating Metrics...")

        # Get Probabilities
        probs_val = model.predict_proba(X_val)[:, 1]

        # A) Calculate Focal Loss
        val_focal_loss = calculate_focal_loss(y_val, probs_val)

        # B) Calculate Optimal Threshold (F0.5 score)
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
        beta = 0.5
        fbeta_scores = (1 + beta ** 2) * (precisions * recalls) / ((beta ** 2 * precisions) + recalls + 1e-8)
        best_idx = np.argmax(fbeta_scores)
        best_threshold = thresholds[best_idx]

        # C) Calculate Precision/Recall at that threshold
        preds_val = (probs_val >= best_threshold).astype(int)
        val_precision = precision_score(y_val, preds_val)
        val_recall = recall_score(y_val, preds_val)

        logger.info(f"Best Threshold: {best_threshold:.4f}")
        logger.info(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")

        # 4. Log Final Summary to W&B
        wandb.log({
            "val_focal_loss": val_focal_loss,
            "best_threshold": best_threshold,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f05": fbeta_scores[best_idx],
            # Log the PR Curve for visualization
            "pr_curve": wandb.plot.pr_curve(
                y_val,
                np.stack([1 - probs_val, probs_val], axis=1),
                labels=["Forest", "Deforestation"]
            )
        })

    wandb.finish()