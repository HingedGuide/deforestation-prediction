import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc

from deforestation_predictor.training.dataset import DeforestationDataset
from deforestation_predictor.models.architectures import Simple3DCNN, ResUNet, ConvLSTM, SimpleViT3D
from deforestation_predictor.training.loss import FocalLoss
from deforestation_predictor.utils.logger import setup_logger

# Initialize logger for this module
logger = setup_logger(__name__, log_file="training_experiment.log")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    # tqdm for progress bar, but we rely on logger for persistent records
    pbar = tqdm(loader, desc="Training", leave=False)

    for X, y in pbar:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Runs validation and calculates metrics (PR-AUC).
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

        # Calculate probabilities for class 1 (Deforestation)
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]

        # Mask ignore labels (usually 2) for metric calculation
        mask = y != 2

        # Flatten and store
        if mask.sum() > 0:
            all_preds.append(probs[mask].cpu().numpy())
            all_targets.append(y[mask].cpu().numpy())

    avg_loss = total_loss / len(loader)

    # If no valid pixels found (rare edge case), return 0 AUC
    if not all_targets:
        logger.warning("No valid pixels found in validation set for metrics.")
        return avg_loss, 0.0

    # Concatenate all batches
    y_true = np.concatenate(all_targets)
    y_scores = np.concatenate(all_preds)

    # Calculate Precision-Recall AUC (Main metric for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    return avg_loss, pr_auc


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Training Loop for Deforestation Prediction")
    parser.add_argument("--data_root", type=str, required=True, help="Path to processed 3D data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    # RQ Parameters
    parser.add_argument("--context_months", type=int, default=12, help="Input window length (RQ2)")
    # Update help text to include resunet
    parser.add_argument("--model_type", type=str, default="3dcnn", choices=["3dcnn", "resunet", "convlstm", "vit"], help="Architecture (RQ1)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models")

    args = parser.parse_args()

    # ... setup logger and directories ...
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ... logging info ...

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

        logger.info(
            f"Input Shape detected: Channels={in_channels}, Time={time_depth}, H={sample_X.shape[2]}, W={sample_X.shape[3]}")

        # --- Updated Model Selection Logic ---
        if args.model_type == "3dcnn":
            model = Simple3DCNN(in_channels=in_channels, time_depth=time_depth).to(device)

        elif args.model_type == "resunet":
            model = ResUNet(in_channels=in_channels, time_depth=time_depth).to(device)

        elif args.model_type == "convlstm":
            # Using the new class
            model = ConvLSTM(in_channels=in_channels, time_depth=time_depth).to(device)

        elif args.model_type == "vit":
            # ViT requires careful sizing. Assuming 64x64 patches.
            # If your PATCH_SIZE changes, you might need to adjust parameters here.
            model = SimpleViT3D(in_channels=in_channels, time_depth=time_depth, img_size=64).to(device)

        else:
            raise ValueError(f"Model type '{args.model_type}' not implemented.")

        logger.info(f"Model {args.model_type} initialized successfully.")

        # ... Optimizer, Loss, Training Loop (keep exactly as is) ...
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)

        best_auc = 0.0

        for epoch in range(args.epochs):
             # ... (keep existing loop content) ...
            logger.info(f"Epoch {epoch + 1}/{args.epochs} started...")

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            logger.info(
                f"Epoch {epoch + 1} Summary: "
                f"Train Loss={train_loss:.4f} | "
                f"Val Loss={val_loss:.4f} | "
                f"Val PR-AUC={val_auc:.4f}"
            )

            # Checkpoint saving
            if val_auc > best_auc:
                best_auc = val_auc
                save_path = save_dir / "best_model.pth"
                torch.save(model.state_dict(), save_path)
                logger.info(f"--> New best model saved to {save_path} (AUC: {best_auc:.4f})")

    except Exception as e:
        logger.exception("An error occurred during the experiment execution.")
        raise e

if __name__ == "__main__":
    main()