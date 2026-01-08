import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

# Import your project modules
from deforestation_predictor.training.dataset import DeforestationDataset
from deforestation_predictor.models.architectures import (
    ResUNet, ResUNet3D, ViViTSegmentation, ConvLSTM3D
)

def get_model(model_type, in_channels, time_depth, device):
    """
    Factory function to initialize the model architecture based on the type name.
    """
    if model_type == "3dcnn":
        return Simple3DCNN(in_channels=in_channels, time_depth=time_depth).to(device)
    elif model_type == "resunet":
        return ResUNet(in_channels=in_channels, time_depth=time_depth).to(device)
    elif model_type == "convlstm":
        return ConvLSTM(in_channels=in_channels, time_depth=time_depth).to(device)
    elif model_type == "vivit":
        return ViViTSegmentation(in_channels=in_channels, time_depth=time_depth).to(device)
    elif model_type == "convlstm3d":
        return ConvLSTM3D(in_channels=in_channels, time_depth=time_depth).to(device)
    elif model_type == "resunet3d":
        return ResUNet3D(in_channels=in_channels, time_depth=time_depth).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def visualize_batch(model, loader, device, output_dir, threshold=0.5, num_images=5):
    """
    Runs inference on a few samples and saves the comparison plots.
    
    Args:
        model: The trained PyTorch model.
        loader: DataLoader for the test set.
        device: 'cuda' or 'cpu'.
        output_dir: Path to save the images.
        threshold: Probability threshold for binary classification.
        num_images: Number of images to generate.
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations in: {output_dir}")

    # Get a single batch
    try:
        X_batch, y_batch = next(iter(loader))
    except StopIteration:
        print("DataLoader is empty.")
        return

    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        logits = model(X_batch)
        # Apply softmax to get probabilities for class 1 (deforestation)
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
    
    # Loop through the batch to create plots
    for i in range(min(num_images, len(X_batch))):
        
        # Prepare data for plotting
        # 1. Prediction (Binary mask based on threshold)
        pred_mask = (probs[i] > threshold).cpu().numpy().astype(int)
        
        # 2. Ground Truth
        gt_mask = y_batch[i].cpu().numpy().astype(int)
        
        # 3. Input reference (Optional: take 1st channel of last time step)
        # Shape is [C, T, H, W], we take channel 0, last time step -1
        input_ref = X_batch[i, 0, -1, :, :].cpu().numpy()
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot Input Reference
        axes[0].imshow(input_ref, cmap='gray')
        axes[0].set_title("Input (Channel 0, Last Month)")
        axes[0].axis('off')

        # Plot Ground Truth
        # GT typically has 0=Forest, 1=Deforestation, 2=Ignore
        cmap_gt = plt.cm.get_cmap("viridis", 3) 
        axes[1].imshow(gt_mask, cmap=cmap_gt, vmin=0, vmax=2)
        axes[1].set_title("Ground Truth\n(Yel=Ignore, Purp=For, Teal=Def)")
        axes[1].axis('off')

        # Plot Prediction
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title(f"Prediction (Threshold {threshold})")
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = output_dir / f"sample_{i}_viz.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Model Predictions")
    parser.add_argument("--data_root", type=str, required=True, help="Path to processed 3D data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth model checkpoint")
    parser.add_argument("--model_type", type=str, required=True, choices=["3dcnn", "resunet", "convlstm", "vivit", "convlstm3d", "resunet3d"])
    parser.add_argument("--context_months", type=int, default=12, help="Context length used during training")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Where to save images")
    parser.add_argument("--num_samples", type=int, default=10, help="How many images to generate")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Dataset (Test split)
    # We use the dataset class to handle .npz loading and temporal slicing
    test_ds = DeforestationDataset(args.data_root, "test", context_length=args.context_months)
    test_loader = DataLoader(test_ds, batch_size=args.num_samples, shuffle=True)
    
    if len(test_ds) == 0:
        print("No samples found in test set.")
        return

    # 2. Determine Input Shapes from a sample
    sample_X, _ = test_ds[0]
    in_channels = sample_X.shape[0]
    time_depth = sample_X.shape[1]
    
    print(f"Detected Input Shape: Channels={in_channels}, Time={time_depth}")

    # 3. Load Model
    model = get_model(args.model_type, in_channels, time_depth, device)
    
    # Load weights
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    
    # 4. Visualize
    visualize_batch(
        model, 
        test_loader, 
        device, 
        output_dir=args.output_dir, 
        num_images=args.num_samples
    )

if __name__ == "__main__":
    main()