import argparse
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, fbeta_score

from deforestation_predictor.training.dataset import DeforestationDataset
from deforestation_predictor.models.architectures import Simple3DCNN, ResUNet, ConvLSTM, ViViTSegmentation
from deforestation_predictor.training.loss import FocalLoss
from deforestation_predictor.utils.logger import setup_logger

# Initialize logger for this module
logger = setup_logger(__name__, log_file="training_experiment.log")


def log_validation_visuals(model, loader, device, epoch, num_samples=4):
    """
    Logs input, ground truth, and prediction images to W&B.
    Since inputs are 3D [C, T, H, W], we visualize the LAST time step.
    """
    model.eval()
    images_to_log = []

    try:
        X, y = next(iter(loader))
    except StopIteration:
        return

    X, y = X.to(device), y.to(device)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]

    for i in range(min(num_samples, len(X))):
        img_t = X[i, 0, -1, :, :].cpu().numpy()
        img_min, img_max = img_t.min(), img_t.max()
        if img_max > img_min:
            img_t = (img_t - img_min) / (img_max - img_min)
        else:
            img_t = np.zeros_like(img_t)

        gt_t = y[i].cpu().numpy()
        pred_t = probs[i].cpu().numpy()

        images_to_log.append(wandb.Image(
            img_t,
            masks={
                "predictions": {
                    "mask_data": (pred_t > 0.5).astype(int),
                    "class_labels": {0: "Forest", 1: "Deforestation"}
                },
                "ground_truth": {
                    "mask_data": gt_t.astype(int),
                    "class_labels": {0: "Forest", 1: "Deforestation", 2: "Ignore"}
                }
            },
            caption=f"Epoch {epoch} - Sample {i}"
        ))

    wandb.log({"Visual Results": images_to_log}, step=epoch)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Training", leave=False)

    for step, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1, :, :]
            mask = y != 2
            if mask.sum() > 0:
                all_preds.append(probs[mask].cpu().numpy())
                all_targets.append(y[mask].cpu().numpy())

        if step % 10 == 0:
            wandb.log({"train_batch_loss": loss.item()})
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)

    if not all_targets:
        train_auc = 0.0
    else:
        y_true = np.concatenate(all_targets)
        y_scores = np.concatenate(all_preds)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        train_auc = auc(recall, precision)

    return avg_loss, train_auc


@torch.no_grad()
def validate(model, loader, criterion, device, return_preds=False):
    """
    Runs validation. If return_preds=True, returns the full probability and target arrays
    for threshold optimization.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X, y in tqdm(loader, desc="Validation", leave=False):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        mask = y != 2

        if mask.sum() > 0:
            all_preds.append(probs[mask].cpu().numpy())
            all_targets.append(y[mask].cpu().numpy())

    avg_loss = total_loss / len(loader)

    if not all_targets:
        logger.warning("No valid pixels found in validation set.")
        return avg_loss, 0.0, None, None

    y_true = np.concatenate(all_targets)
    y_scores = np.concatenate(all_preds)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    if return_preds:
        return avg_loss, pr_auc, y_scores, y_true
    
    return avg_loss, pr_auc


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Training Loop")
    parser.add_argument("--data_root", type=str, required=True, help="Path to processed 3D data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context_months", type=int, default=12, help="Input window length")
    parser.add_argument("--model_type", type=str, default="3dcnn", choices=["3dcnn", "resunet", "convlstm", "vivit"])
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--wandb_project", type=str, default="deforestation-prediction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"{args.model_type}_ctx{args.context_months}",
        config=vars(args)
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 1. Load Datasets
        train_ds = DeforestationDataset(args.data_root, "train", context_length=args.context_months)
        val_ds = DeforestationDataset(args.data_root, "val", context_length=args.context_months)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # 2. Initialize Model
        sample_X, _ = train_ds[0]
        in_channels = sample_X.shape[0]
        time_depth = sample_X.shape[1]

        logger.info(f"Input Shape: C={in_channels}, T={time_depth}, H={sample_X.shape[2]}, W={sample_X.shape[3]}")

        if args.model_type == "3dcnn":
            model = Simple3DCNN(in_channels=in_channels, time_depth=time_depth).to(device)
        elif args.model_type == "resunet":
            model = ResUNet(in_channels=in_channels, time_depth=time_depth).to(device)
        elif args.model_type == "convlstm":
            model = ConvLSTM(in_channels=in_channels, time_depth=time_depth).to(device)
        elif args.model_type == "vivit":
            model = ViViTSegmentation(in_channels=in_channels, time_depth=args.context_months).to(device)
        else:
            raise ValueError(f"Model type '{args.model_type}' not implemented.")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)

        best_auc = 0.0
        filename = f"{args.wandb_run_name or args.model_type}_best.pth"
        best_model_path = save_dir / filename

        # --- Training Loop ---
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch + 1}/{args.epochs} started...")
            train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_pr_auc": train_auc,
                "val_loss": val_loss,
                "val_pr_auc": val_auc
            }, step=epoch + 1)

            log_validation_visuals(model, val_loader, device, epoch + 1)

            logger.info(f"Epoch {epoch + 1}: Train AUC={train_auc:.4f} | Val AUC={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"--> New best model saved (AUC: {best_auc:.4f})")

        # --- Test Evaluation ---
        logger.info("Training Complete. Starting Test Evaluation...")
        
        # 1. Load Best Model
        model.load_state_dict(torch.load(best_model_path))
        
        # 2. Load Test Data
        try:
            test_ds = DeforestationDataset(args.data_root, "test", context_length=args.context_months)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        except Exception:
            logger.error("Test set not found or empty. Skipping test evaluation.")
            return

        # 3. Find Optimal Threshold on Validation Set
        # We re-run validation on the best model to get predictions
        _, _, val_probs, val_targets = validate(model, val_loader, criterion, device, return_preds=True)
        
        prec, rec, thresholds = precision_recall_curve(val_targets, val_probs)
        beta = 0.5
        fbeta = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec + 1e-8)
        best_idx = np.argmax(fbeta)
        best_threshold = thresholds[best_idx]
        val_f05 = fbeta[best_idx]
        
        logger.info(f"Optimal Threshold found on Val: {best_threshold:.4f} (Val F0.5: {val_f05:.4f})")

        # 4. Evaluate on Test Set
        test_loss, test_auc, test_probs, test_targets = validate(model, test_loader, criterion, device, return_preds=True)
        
        # Apply threshold
        test_preds = (test_probs >= best_threshold).astype(int)
        
        test_precision = precision_score(test_targets, test_preds)
        test_recall = recall_score(test_targets, test_preds)
        test_f05 = fbeta_score(test_targets, test_preds, beta=0.5)

        logger.info("="*30)
        logger.info(f"FINAL TEST RESULTS (at threshold {best_threshold:.4f})")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test PR-AUC: {test_auc:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F0.5 Score: {test_f05:.4f}")
        logger.info("="*30)

        # Log to W&B
        wandb.log({
            "test_loss": test_loss,
            "test_pr_auc": test_auc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f05": test_f05,
            "optimal_threshold": best_threshold,
            "test_pr_curve": wandb.plot.pr_curve(
                test_targets, 
                np.stack([1-test_probs, test_probs], axis=1), 
                labels=["Forest", "Deforestation"]
            )
        })

    except Exception as e:
        logger.exception("An error occurred during execution.")
        raise e
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()