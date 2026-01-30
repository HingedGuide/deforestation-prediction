import argparse
import math
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, fbeta_score

# Ensure these match your actual file structure
from deforestation_predictor.training.dataset import DeforestationDataset
from deforestation_predictor.models.architectures import ResUNet, ResUNet3D, ViViTSegmentation, ConvLSTM3D
from deforestation_predictor.training.loss import WeightedFocalLoss, CombinedLoss
from deforestation_predictor.utils.logger import setup_logger

# Initialize logger for this module
logger = setup_logger(__name__, log_file="training_experiment.log")


def log_validation_visuals(model, loader, device, epoch, num_samples=4):
    """
    Logs input, ground truth, and prediction images to W&B.
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
        # Handle 1D channel output
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits).squeeze(1)
        else:
            probs = torch.sigmoid(logits[:, 0]) 

    for i in range(min(num_samples, len(X))):
        # VISUALIZATION FIX: Handle 4D or 5D input
        # If [B, C, T, H, W], take last time step. If [B, C, H, W], take whole image.
        if X.ndim == 5:
            img_t = X[i, 0, -1, :, :].cpu().numpy()
        else:
            img_t = X[i, 0, :, :].cpu().numpy()

        # Normalize for display
        img_min, img_max = img_t.min(), img_t.max()
        if img_max > img_min:
            img_t = (img_t - img_min) / (img_max - img_min)
        else:
            img_t = np.zeros_like(img_t)

        gt_t = y[i].cpu().numpy()
        pred_t = probs[i].cpu().numpy()
        
        # FIX: Clean GT for visualization (map 255/garbage to 2 for 'Ignore')
        gt_viz = gt_t.copy()
        gt_viz[(gt_viz != 0) & (gt_viz != 1)] = 2

        images_to_log.append(wandb.Image(
            img_t,
            masks={
                "predictions": {
                    "mask_data": (pred_t > 0.5).astype(int),
                    "class_labels": {0: "Forest", 1: "Deforestation"}
                },
                "ground_truth": {
                    "mask_data": gt_viz.astype(int),
                    "class_labels": {0: "Forest", 1: "Deforestation", 2: "Ignore"}
                }
            },
            caption=f"Epoch {epoch} - Sample {i}"
        ))

    wandb.log({"Visual Results": images_to_log}, step=epoch)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, args):
    model.train()
    total_loss = 0.0

    total_steps = len(loader) * args.epochs
    warmup_steps = len(loader) * args.warmup_epochs

    # --- Custom Scheduler Logic (Kept as is) ---
    warmup_schedule = np.linspace(0, args.lr, warmup_steps)
    iters = np.arange(total_steps - warmup_steps)
    cosine_schedule = np.array([
        0 + 0.5 * (args.lr - 0) * (1 + math.cos(math.pi * t / (total_steps - warmup_steps)))
        for t in iters
    ])
    lr_schedule = np.concatenate((warmup_schedule, cosine_schedule))

    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Training", leave=False)

    for step, (X, y) in enumerate(pbar):
        global_step = epoch * len(loader) + step
        if global_step < len(lr_schedule):
            current_lr = lr_schedule[global_step]
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        X, y = X.to(device), y.to(device)

        # --- FIX: STRICT MASKING ---
        valid_mask = (y == 0) | (y == 1)
        
        # --- IMPLEMENTATIE WEIGHTING (HET BREEKIJZER) ---
        # Maak een mask met gewichten: Bos=1.0, Kap=10.0
        weight_mask = torch.zeros_like(y, dtype=torch.float, device=device)
        weight_mask[y == 0] = 1.0
        weight_mask[y == 1] = 10.0  # Forceer focus op ontbossing!
        weight_mask[~valid_mask] = 0.0

        optimizer.zero_grad()
        logits = model(X)
        logits = logits.squeeze(1)

        # --- FIX: CLEAN TARGETS FOR LOSS ---
        y_clean = y.clone()
        y_clean[~valid_mask] = 0 

        # Forward loss with weight mask
        loss = criterion(logits, y_clean, weight_mask)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # --- FIX: METRICS ---
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            if valid_mask.sum() > 0:
                all_preds.append(probs[valid_mask].cpu().numpy())
                all_targets.append(y[valid_mask].cpu().numpy())

        if step % 10 == 0:
            wandb.log({"train_batch_loss": loss.item()})
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)

    # Metrics calculation
    train_auc = 0.0
    train_f05 = 0.0
    train_precision = 0.0
    train_recall = 0.0

    if all_targets:
        y_true = np.concatenate(all_targets)
        y_scores = np.concatenate(all_preds)
        
        if len(np.unique(y_true)) < 2:
            train_auc = 0.5
        else:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            train_auc = auc(recall, precision)
            
            # --- Calculate Max F0.5 Score for Training ---
            with np.errstate(divide='ignore', invalid='ignore'):
                beta = 0.5
                numerator = (1 + beta**2) * precision * recall
                denominator = (beta**2 * precision) + recall
                fbeta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
            
            # Find the index of the best F0.5 score
            best_idx = np.argmax(fbeta)
            train_f05 = fbeta[best_idx]
            
            # Get Precision and Recall at that specific threshold
            train_precision = precision[best_idx]
            train_recall = recall[best_idx]

    return avg_loss, train_auc, train_f05, train_precision, train_recall


@torch.no_grad()
def validate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X, y in tqdm(loader, desc="Validation", leave=False):
        X, y = X.to(device), y.to(device)
        
        # --- FIX: STRICT MASKING & WEIGHTING ---
        valid_mask = (y == 0) | (y == 1)
        
        # Consistent weighting in validation (for fair loss comparison)
        weight_mask = torch.zeros_like(y, dtype=torch.float, device=device)
        weight_mask[y == 0] = 1.0
        weight_mask[y == 1] = 10.0
        weight_mask[~valid_mask] = 0.0
        
        logits = model(X)
        logits = logits.squeeze(1)

        y_clean = y.clone()
        y_clean[~valid_mask] = 0

        loss = criterion(logits, y_clean, weight_mask)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)

        if valid_mask.sum() > 0:
            all_preds.append(probs[valid_mask].cpu().numpy())
            all_targets.append(y[valid_mask].cpu().numpy())

    avg_loss = total_loss / len(loader)

    if not all_targets:
        logger.warning("No valid pixels found in validation set.")
        if return_preds: return avg_loss, 0.0, 0.0, 0.0, 0.0, None, None
        return avg_loss, 0.0, 0.0, 0.0, 0.0

    y_true = np.concatenate(all_targets)
    y_scores = np.concatenate(all_preds)
    
    val_auc = 0.5
    val_f05 = 0.0
    val_precision = 0.0
    val_recall = 0.0

    if len(np.unique(y_true)) >= 2:
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        val_auc = auc(recall, precision)

        # --- Calculate Max F0.5 Score for Validation ---
        with np.errstate(divide='ignore', invalid='ignore'):
            beta = 0.5
            numerator = (1 + beta**2) * precision * recall
            denominator = (beta**2 * precision) + recall
            fbeta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        best_idx = np.argmax(fbeta)
        val_f05 = fbeta[best_idx]
        
        # Get Precision/Recall at the optimal threshold
        val_precision = precision[best_idx]
        val_recall = recall[best_idx]

    if return_preds:
        return avg_loss, val_auc, val_f05, val_precision, val_recall, y_scores, y_true
    
    return avg_loss, val_auc, val_f05, val_precision, val_recall

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Training Loop")
    parser.add_argument("--data_root", type=str, required=True, help="Path to processed 3D data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--epoch_size", type=int, default=50000, help="Number of samples per epoch")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--context_months", type=int, default=12, help="Input window length")
    parser.add_argument("--model_type", type=str, default="resunet", choices=["resunet", "vivit", "convlstm3d", "resunet3d"])
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--wandb_project", type=str, default="deforestation-prediction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument('--mode', type=str, default='snapshot', choices=['sequence', 'snapshot'],
                        help="Training mode: 'sequence' for 3D models, 'snapshot' for 2D models.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")

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
        train_ds = DeforestationDataset(
            args.data_root, 
            "train", 
            context_length=args.context_months, 
            mode=args.mode, 
            epoch_size=args.epoch_size)
        
        val_ds = DeforestationDataset(
            args.data_root,
            "val", 
            context_length=args.context_months, 
            mode=args.mode, epoch_size=args.epoch_size // 2
            )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # 2. Initialize Model
        sample_X, _ = train_ds[0]
        in_channels = sample_X.shape[0]
        
        # Determine time depth based on mode
        if args.mode == 'snapshot':
            time_depth = 1 
        else: # 'sequence' mode
            time_depth = args.context_months if args.context_months else 12

        logger.info(f"Input Shape: C={in_channels}, T={time_depth}, H={sample_X.shape[-2]}, W={sample_X.shape[-1]}")

        # Model Factory
        if args.model_type == "resunet":
            # 2D Model
            model = ResUNet(in_channels=in_channels, time_depth=time_depth, num_classes=1).to(device)
        elif args.model_type == "vivit":
            model = ViViTSegmentation(in_channels=in_channels, time_depth=time_depth, num_classes=1).to(device)
        elif args.model_type == "convlstm3d":
            model = ConvLSTM3D(in_channels=in_channels, time_depth=time_depth, num_classes=1).to(device)
        elif args.model_type == "resunet3d":
            model = ResUNet3D(in_channels=in_channels, time_depth=time_depth, num_classes=1).to(device)
        else:
            raise ValueError(f"Model type '{args.model_type}' not implemented.")

        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Select Loss Function
        # criterion = WeightedFocalLoss(alpha=0.25, gamma=2.0).to(device)
        criterion = CombinedLoss().to(device)

        best_f05 = 0.0 
        filename = f"{args.wandb_run_name or args.model_type}_best.pth"
        best_model_path = save_dir / filename

        # --- Training Loop ---
        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch + 1}/{args.epochs} started...")
            
            # Unpack new return values
            train_loss, train_auc, train_f05, train_prec, train_rec = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch, args
            )
            val_loss, val_auc, val_f05, val_prec, val_rec = validate(
                model, val_loader, criterion, device
            )

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_pr_auc": train_auc,
                "train_f05": train_f05, 
                "train_precision": train_prec,
                "train_recall": train_rec,
                "val_loss": val_loss,
                "val_pr_auc": val_auc,
                "val_f05": val_f05,
                "val_precision": val_prec,
                "val_recall": val_rec
            }, step=epoch + 1)

            log_validation_visuals(model, val_loader, device, epoch + 1)

            logger.info(f"Epoch {epoch + 1}: Val F0.5={val_f05:.4f} | Prec={val_prec:.4f} | Rec={val_rec:.4f}")

            # Save based on F0.5 score
            if val_f05 > best_f05:
                best_f05 = val_f05
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"--> New best model saved (F0.5: {best_f05:.4f})")

        # --- Test Evaluation ---
        logger.info("Training Complete. Starting Test Evaluation...")
        
        # 1. Load Best Model
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path))
        else:
            logger.warning("No best model found, using last epoch model.")
        
        # 2. Load Test Data
        try:
            test_ds = DeforestationDataset(args.data_root, "test", context_length=args.context_months, mode=args.mode)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        except Exception:
            logger.error("Test set not found or empty. Skipping test evaluation.")
            return

        # 3. Find Optimal Threshold on Validation Set
        # We re-run validation on the best model to get predictions
        _, _, _, _, _, val_probs, val_targets = validate(model, val_loader, criterion, device, return_preds=True)
        
        if len(np.unique(val_targets)) > 1:
            prec, rec, thresholds = precision_recall_curve(val_targets, val_probs)
            beta = 0.5
            fbeta = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec + 1e-8)
            best_idx = np.argmax(fbeta)
            best_threshold = thresholds[best_idx]
            val_f05 = fbeta[best_idx]
        else:
            best_threshold = 0.5
            val_f05 = 0.0
        
        logger.info(f"Optimal Threshold found on Val: {best_threshold:.4f} (Val F0.5: {val_f05:.4f})")

        # 4. Evaluate on Test Set
        test_loss, test_auc, _, _, _, test_probs, test_targets = validate(model, test_loader, criterion, device, return_preds=True)
        
        # Apply threshold
        test_preds = (test_probs >= best_threshold).astype(int)
        
        test_precision = precision_score(test_targets, test_preds, zero_division=0)
        test_recall = recall_score(test_targets, test_preds, zero_division=0)
        test_f05 = fbeta_score(test_targets, test_preds, beta=0.5, zero_division=0)

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