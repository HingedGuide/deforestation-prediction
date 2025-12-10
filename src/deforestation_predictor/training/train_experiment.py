import argparse
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc

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

    # Get one batch
    # We use next(iter()) which might be slow if loader is heavy, but okay for logging
    try:
        X, y = next(iter(loader))
    except StopIteration:
        return

    X, y = X.to(device), y.to(device)

    with torch.no_grad():
        logits = model(X)
        # Prob of deforestation (Class 1)
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]

    # Select a few samples from the batch
    for i in range(min(num_samples, len(X))):
        # 1. Extract Input (Last time step)
        # X shape: [C, T, H, W]. We take T=-1.
        # Assuming channel 0 is a visible band or precipitation. Adjust index if needed.
        img_t = X[i, 0, -1, :, :].cpu().numpy()

        # Normalize for display [0, 1]
        img_min, img_max = img_t.min(), img_t.max()
        if img_max > img_min:
            img_t = (img_t - img_min) / (img_max - img_min)
        else:
            img_t = np.zeros_like(img_t)

        # 2. Ground Truth
        gt_t = y[i].cpu().numpy()

        # 3. Prediction
        pred_t = probs[i].cpu().numpy()

        # Create W&B Image with masks
        images_to_log.append(wandb.Image(
            img_t,
            masks={
                "predictions": {
                    "mask_data": (pred_t > 0.5).astype(int),
                    "class_labels": {0: "Forest", 1: "Deforestation"}
                },
                "ground_truth": {
                    # W&B expects integer masks
                    "mask_data": gt_t.astype(int),
                    "class_labels": {0: "Forest", 1: "Deforestation", 2: "Ignore"}
                }
            },
            caption=f"Epoch {epoch} - Sample {i}"
        ))

    wandb.log({"Visual Results": images_to_log}, step=epoch)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Runs one epoch of training, calculating Loss and PR-AUC.
    """
    model.train()
    total_loss = 0.0

    # Lists to store predictions for PR-AUC calculation
    all_preds = []
    all_targets = []

    # tqdm for progress bar
    pbar = tqdm(loader, desc="Training", leave=False)

    for step, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # --- Metric Collection ---
        # We must detach to prevent memory leaks (keeping graph history)
        with torch.no_grad():
            # Get probabilities for class 1
            probs = torch.softmax(logits, dim=1)[:, 1, :, :]

            # Mask ignore labels (usually 2)
            mask = y != 2

            if mask.sum() > 0:
                all_preds.append(probs[mask].cpu().numpy())
                all_targets.append(y[mask].cpu().numpy())

        # Log batch loss to W&B frequently
        if step % 10 == 0:
            wandb.log({"train_batch_loss": loss.item()})

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)

    # --- Calculate Training PR-AUC ---
    if not all_targets:
        train_auc = 0.0
    else:
        y_true = np.concatenate(all_targets)
        y_scores = np.concatenate(all_preds)

        # Calculate Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        train_auc = auc(recall, precision)

    return avg_loss, train_auc


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
    parser.add_argument("--model_type", type=str, default="3dcnn", choices=["3dcnn", "resunet", "convlstm", "vivit"],
                        help="Architecture (RQ1)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models")

    # W&B Arguments
    parser.add_argument("--wandb_project", type=str, default="deforestation-prediction", help="WandB Project Name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB Entity")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Specific name for this run")

    args = parser.parse_args()

    # --- Initialize W&B ---
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

        elif args.model_type == "vivit":
            # New Factorized ViViT
            logger.info("Initializing Factorized ViViT model...")
            model = ViViTSegmentation(
                in_channels=in_channels,
                time_depth=args.context_months,
                img_size=64,
                patch_size=8,
                embed_dim=128
            ).to(device)

        else:
            raise ValueError(f"Model type '{args.model_type}' not implemented.")

        # Watch gradients
        wandb.watch(model, log="all", log_freq=100)

        logger.info(f"Model {args.model_type} initialized successfully.")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)

        best_auc = 0.0

        for epoch in range(args.epochs):
            logger.info(f"Epoch {epoch + 1}/{args.epochs} started...")

            # Run Training
            train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

            # Run Validation
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            # Log Metrics
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_pr_auc": train_auc,
                "val_loss": val_loss,
                "val_pr_auc": val_auc
            }, step=epoch + 1)

            # Log Visuals
            log_validation_visuals(model, val_loader, device, epoch + 1)

            logger.info(
                f"Epoch {epoch + 1} Summary: "
                f"Train Loss={train_loss:.4f} | Train AUC={train_auc:.4f} | "
                f"Val Loss={val_loss:.4f} | Val AUC={val_auc:.4f}"
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
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()