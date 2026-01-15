import argparse
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, fbeta_score

# Dataset
from deforestation_predictor.training.dataset import DeforestationDataset
from deforestation_predictor.models.architectures import ResUNet, ResUNet3D, ViViTSegmentation, ConvLSTM3D
from deforestation_predictor.training.loss import WeightedFocalLoss
from deforestation_predictor.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__, log_file="training_experiment.log")

def find_best_f05_threshold(y_true: np.ndarray, y_probs: np.ndarray):
    """
    Finds the probability threshold that maximizes the F0.5 score.
    
    Args:
        y_true: Flattened numpy array of ground truth labels (0 or 1).
        y_probs: Flattened numpy array of predicted probabilities.
        
    Returns:
        best_threshold (float): The threshold that gives the highest F0.5.
        best_score (float): The maximum F0.5 score achieved.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Formula: (1 + beta^2) * (P * R) / ((beta^2 * P) + R) where beta = 0.5
    beta_sq = 0.5 ** 2
    
    # Handle division by zero safely
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = (1 + beta_sq) * (precision * recall)
        denominator = (beta_sq * precision) + recall
        f05_scores = numerator / denominator
        f05_scores[np.isnan(f05_scores)] = 0.0
    
    # Find the index of the best score
    best_idx = np.argmax(f05_scores)
    
    # Handle edge case where thresholds array is shorter than precision/recall arrays
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    best_score = f05_scores[best_idx]
    
    return best_threshold, best_score

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch using WeightedFocalLoss.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)

    for step, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        
        # Laura's setup typically expects float targets for BCE-based Focal Loss
        # Ensure y is (B, H, W) and logits are (B, C, H, W) or (B, H, W)
        # Assuming binary segmentation where class 1 is deforestation.
        
        # If model outputs 1 channel (B, 1, H, W), remove channel dim for loss if needed
        # OR if model outputs 2 channels (B, 2, H, W), pick channel 1 or use CrossEntropy logic.
        
        # ADAPTATION: Assuming standard BCE-like behavior for binary segmentation
        if logits.shape[1] == 1:
             loss = criterion(logits.squeeze(1), y.float())
        else:
            # If 2 channels, often we take the 2nd channel for binary loss or use CE
            # Here assuming binary output setup common in these unets
             loss = criterion(logits[:, 1, :, :], y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Store predictions for AUC calculation
        with torch.no_grad():
            # Apply sigmoid/softmax to get probability of positive class
            if logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=1)[:, 1, :, :]
            else:
                probs = torch.sigmoid(logits).squeeze(1)
            
            mask = y != 2 # 2 is often the ignore index in these datasets
            if mask.sum() > 0:
                all_preds.append(probs[mask].cpu().numpy())
                all_targets.append(y[mask].cpu().numpy())

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
def validate(model, dataloader, criterion, device):
    """
    Validates the model. 
    Returns loss and raw arrays for F0.5 optimization.
    """
    model.eval()
    val_loss = 0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            # Loss Calculation (matching training logic)
            if outputs.shape[1] == 1:
                 loss = criterion(outputs.squeeze(1), targets.float())
                 probs = torch.sigmoid(outputs).squeeze(1)
            else:
                 loss = criterion(outputs[:, 1, :, :], targets.float())
                 probs = torch.softmax(outputs, dim=1)[:, 1, :, :]
            
            val_loss += loss.item()
            
            # Store predictions and targets on CPU
            # Filter ignore index (2)
            mask = targets != 2
            if mask.sum() > 0:
                all_probs.append(probs[mask].cpu().numpy())
                all_targets.append(targets[mask].cpu().numpy())

    if not all_probs:
        return val_loss, np.array([]), np.array([])

    # Concatenate all batches
    all_probs = np.concatenate(all_probs).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    avg_loss = val_loss / len(dataloader)
    
    return avg_loss, all_probs, all_targets


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Training Loop (Thesis Setup)")
    parser.add_argument("--data_root", type=str, required=True, help="Path to processed 3D data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context_months", type=int, default=12)
    parser.add_argument("--model_type", type=str, default="resunet3d")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default="deforestation-prediction")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"{args.model_type}_ctx{args.context_months}_F05",
        config=vars(args)
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Datasets
    logger.info("Loading datasets...")
    train_ds = DeforestationDataset(args.data_root, "train", context_length=args.context_months)
    val_ds = DeforestationDataset(args.data_root, "val", context_length=args.context_months)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Initialize Model
    sample_X, _ = train_ds[0]
    in_channels = sample_X.shape[0]
    time_depth = sample_X.shape[1] 
    
    logger.info(f"Model: {args.model_type} | Input: C={in_channels}, T={time_depth}")

    if args.model_type == "resunet3d":
        model = ResUNet3D(in_channels=in_channels, time_depth=time_depth).to(device)
    elif args.model_type == "convlstm3d":
        model = ConvLSTM3D(in_channels=in_channels, time_depth=time_depth).to(device)
    elif args.model_type == "vivit":
        model = ViViTSegmentation(in_channels=in_channels, time_depth=time_depth).to(device)
    elif args.model_type == "resunet":
        # 2D baseline: flatten time into channels usually handled inside model or wrapper
        model = ResUNet(in_channels=in_channels * time_depth).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Define Loss (Laura's Configuration)
    # Using WeightedFocalLoss exclusively. 
    # alpha=0.25 typically means weighting the positive class higher (conceptually), 
    # or it is the weight for class 0 depending on implementation. 
    # Laura uses alpha=0.25, gamma=2.0.
    criterion = WeightedFocalLoss(alpha=0.25, gamma=2.0).to(device)

    # Initialize tracking variables
    best_val_f05 = 0.0 
    filename = f"{args.wandb_run_name or args.model_type}_best.pth"
    best_model_path = save_dir / filename

    logger.info("Starting training...")

    # --- Training Loop ---
    for epoch in range(args.epochs):
        
        # Train Step
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validation Step
        val_loss, val_probs, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Calculate metrics & Optimal Threshold
        if len(val_targets) > 0:
            best_thresh, val_f05 = find_best_f05_threshold(val_targets, val_probs)
            
            p, r, _ = precision_recall_curve(val_targets, val_probs)
            val_auc = auc(r, p)
        else:
            best_thresh, val_f05, val_auc = 0.5, 0.0, 0.0

        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F0.5: {val_f05:.4f} (Thresh: {best_thresh:.2f})")
    
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_f05": val_f05,
            "val_best_threshold": best_thresh
        })

        # Save Best Model (Based on F0.5 to match Thesis goals)
        if val_f05 > best_val_f05:
            best_val_f05 = val_f05
            logger.info(f"New best F0.5 score: {best_val_f05:.4f}. Saving model...")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f05': best_val_f05,
                'best_threshold': best_thresh, # CRITICAL: Save threshold for inference
            }, best_model_path)

    # --- Test Evaluation ---
    logger.info("Training Complete. Starting Test Evaluation...")
    
    if not best_model_path.exists():
        logger.warning("No best model file found. Skipping evaluation.")
        return

    # 1. Load Best Model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimal_threshold = checkpoint.get('best_threshold', 0.5) # Retrieve the threshold
    
    logger.info(f"Loaded best model from Epoch {checkpoint['epoch']} with Val F0.5: {checkpoint['best_f05']:.4f}")
    logger.info(f"Using saved optimal threshold from validation: {optimal_threshold:.4f}")

    # 2. Load Test Data
    try:
        test_ds = DeforestationDataset(args.data_root, "test", context_length=args.context_months)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    except Exception:
        logger.error("Test set not found. Skipping.")
        return

    # 3. Evaluate on Test Set using the SAVED threshold
    test_loss, test_probs, test_targets = validate(
        model, test_loader, criterion, device
    )
    
    if len(test_targets) == 0:
        logger.warning("Test set empty or all ignored.")
        return

    # Apply threshold
    test_preds = (test_probs >= optimal_threshold).astype(int)
    
    test_precision = precision_score(test_targets, test_preds, zero_division=0)
    test_recall = recall_score(test_targets, test_preds, zero_division=0)
    test_f05 = fbeta_score(test_targets, test_preds, beta=0.5, zero_division=0)
    
    # Calculate Test AUC
    prec, rec, _ = precision_recall_curve(test_targets, test_probs)
    test_auc = auc(rec, prec)

    logger.info("="*30)
    logger.info(f"FINAL TEST RESULTS (Threshold: {optimal_threshold:.4f})")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test PR-AUC: {test_auc:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    logger.info(f"Test F0.5 Score: {test_f05:.4f}")
    logger.info("="*30)

    wandb.log({
        "test_loss": test_loss,
        "test_pr_auc": test_auc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f05": test_f05,
        "test_threshold_used": optimal_threshold
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()