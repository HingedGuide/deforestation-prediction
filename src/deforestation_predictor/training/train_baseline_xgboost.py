import numpy as np
import xgboost as xgb
import wandb
from pathlib import Path
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import joblib
import logging
import sys
import gc  # Garbage Collector for memory management

# ------------- CONFIG ------------- #
# Calculate absolute project root relative to this script
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "processed_3d" / "GABON"

IGNORE_LABEL = 2
LOG_FILE = "xgboost_baseline.log"

# Cluster Settings
N_ESTIMATORS = 200  # High number for production
MAX_DEPTH = 6  # Standard depth
VAL_TEST_SAMPLE_RATE = 0.2  # Keep 20% of Val/Test pixels to save RAM (statistically sufficient)


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
    p = np.clip(y_pred_prob, 1e-7, 1 - 1e-7)
    ce_loss = - (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    pt = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    focal_loss = alpha_t * (1 - pt) ** gamma * ce_loss
    return np.mean(focal_loss)


# ------------- DATA LOADING ------------- #
def load_and_flatten_data(split_name, sample_rate=0.1, balanced=True):
    split_dir = DATA_ROOT / split_name

    if not split_dir.exists():
        logger.error(f"Directory not found: {split_dir}")
        return None, None

    files = list(split_dir.glob("*.npz"))

    logger.info(f"Loading {split_name} data from {len(files)} patches...")

    X_list = []
    y_list = []

    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                X_cube = data['X']
                y_mask = data['y']
                V, T, H, W = X_cube.shape

                # Flatten [V, T, H, W] -> [Pixels, Features]
                X_flat = X_cube.transpose(2, 3, 0, 1).reshape(H * W, -1)
                y_flat = y_mask.reshape(-1)

                valid_mask = y_flat != IGNORE_LABEL
                X_flat = X_flat[valid_mask]
                y_flat = y_flat[valid_mask]

                if len(y_flat) == 0: continue

                # --- SAMPLING LOGIC ---
                rng = np.random.default_rng()

                if balanced and split_name == 'train':
                    # TRAINING: Keep all positives, downsample negatives to 1:10 ratio
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
                # ----------------------

                X_list.append(X_flat)
                y_list.append(y_flat)
        except Exception:
            pass

    if not X_list: return None, None
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ------------- MAIN EXECUTION ------------- #
if __name__ == "__main__":

    # 1. Initialize W&B
    wandb.init(project="deforestation-prediction", name="xgboost_production_run", tags=["production", "gpu"])

    # 2. Load Training Data
    logger.info("Step 1: Preparing Training Data...")
    # sample_rate=0.1 here means "Keep 10% of NEGATIVES" (Balancing)
    X_train, y_train = load_and_flatten_data("train", sample_rate=0.1, balanced=True)

    # 3. Load Validation Data
    logger.info("Step 1b: Preparing Validation Data...")
    # sample_rate=0.2 here means "Keep 20% of TOTAL data" (Memory saving)
    X_val, y_val = load_and_flatten_data("val", sample_rate=VAL_TEST_SAMPLE_RATE, balanced=False)

    if X_train is not None and X_val is not None:
        logger.info(f"Step 2: Training XGBoost on GPU ({X_train.shape[0]} samples)...")

        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=0.1,
            n_jobs=-1,
            # --- GPU & Performance ---
            tree_method='hist',
            device='cuda',
            # -------------------------
            eval_metric=["logloss", "aucpr"]
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False  # Keep logs clean, we log to W&B manually below
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
        joblib.dump(model, "xgboost_production.pkl")
        logger.info("Model saved. Cleaning memory...")

        # --- MEMORY CLEANUP ---
        del X_train, y_train, X_val, y_val
        gc.collect()
        # ----------------------

        # 4. Final Evaluation on TEST SET
        logger.info("Step 4: Loading TEST Data...")
        X_test, y_test = load_and_flatten_data("test", sample_rate=VAL_TEST_SAMPLE_RATE, balanced=False)

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
                "test_f05": fbeta_scores[best_idx],
                "test_pr_curve": wandb.plot.pr_curve(
                    y_test,
                    np.stack([1 - probs_test, probs_test], axis=1),
                    labels=["Forest", "Deforestation"]
                )
            })

    wandb.finish()