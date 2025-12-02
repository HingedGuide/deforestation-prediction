import pytest
import numpy as np
import torch
import shutil
from pathlib import Path

# Import the components we want to test
from deforestation_predictor.training.dataset import DeforestationDataset
from deforestation_predictor.models.architectures import Simple3DCNN
from deforestation_predictor.training.loss import FocalLoss


# ---------------------------------------------------------
# Test Dataset Loading & Logic (RQ2)
# ---------------------------------------------------------

@pytest.fixture
def mock_data_root(tmp_path):
    """
    Creates a temporary directory structure with dummy .npz files.
    Structure:
      tmp_path/train/
        sample_0.npz
        sample_1.npz
    """
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    # Create dummy data: 5 channels, 12 months, 64x64 spatial
    # Sample 0
    np.savez(
        train_dir / "sample_0.npz",
        X=np.random.randn(5, 12, 64, 64).astype(np.float32),
        y=np.random.randint(0, 3, (64, 64)).astype(np.int64)
    )
    # Sample 1
    np.savez(
        train_dir / "sample_1.npz",
        X=np.random.randn(5, 12, 64, 64).astype(np.float32),
        y=np.random.randint(0, 3, (64, 64)).astype(np.int64)
    )

    return tmp_path


def test_dataset_initialization(mock_data_root):
    """Test if dataset loads the correct number of files."""
    ds = DeforestationDataset(mock_data_root, split="train")
    assert len(ds) == 2


def test_dataset_shapes(mock_data_root):
    """Test if __getitem__ returns correct tensor shapes."""
    ds = DeforestationDataset(mock_data_root, split="train")
    X, y = ds[0]

    # Expected: X=[5, 12, 64, 64], y=[64, 64]
    assert X.shape == (5, 12, 64, 64)
    assert y.shape == (64, 64)
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)


def test_dataset_rq2_temporal_slicing(mock_data_root):
    """
    Test RQ2 logic: If context_length is provided,
    the dataset should slice the time dimension.
    """
    # We ask for only the last 3 months
    context = 3
    ds = DeforestationDataset(mock_data_root, split="train", context_length=context)
    X, y = ds[0]

    # Original was 12 months, requested 3. Output dim 1 should be 3.
    assert X.shape[1] == context

    # Verify it took the *last* 3 months.
    # We load the raw file to compare.
    raw_data = np.load(mock_data_root / "train" / "sample_0.npz")
    raw_X = raw_data['X']

    # The pytorch tensor should match the raw numpy array's last 3 indices
    expected_slice = raw_X[:, -3:, :, :]

    assert np.allclose(X.numpy(), expected_slice)


def test_dataset_missing_files(tmp_path):
    """Test that ValueError is raised if directory is empty."""
    (tmp_path / "train").mkdir()
    with pytest.raises(ValueError, match="No .npz files found"):
        DeforestationDataset(tmp_path, split="train")


# ---------------------------------------------------------
# Test Model Architecture (RQ1)
# ---------------------------------------------------------

def test_3dcnn_forward_pass():
    """
    Test that the Simple3DCNN accepts input of shape [B, C, T, H, W]
    and outputs [B, num_classes, H, W].
    """
    batch_size = 2
    channels = 4
    time_depth = 6
    height, width = 32, 32
    num_classes = 2

    model = Simple3DCNN(in_channels=channels, time_depth=time_depth, num_classes=num_classes)

    # Create dummy input
    dummy_input = torch.randn(batch_size, channels, time_depth, height, width)

    # Forward pass
    output = model(dummy_input)

    # Check output shape
    assert output.shape == (batch_size, num_classes, height, width)


def test_3dcnn_gradients():
    """Test that gradients are propagated (backward pass works)."""
    model = Simple3DCNN(in_channels=3, time_depth=4)
    dummy_input = torch.randn(2, 3, 4, 32, 32)
    dummy_target = torch.randint(0, 2, (2, 32, 32))

    output = model(dummy_input)
    loss = torch.nn.functional.cross_entropy(output, dummy_target)
    loss.backward()

    # Check if a parameter has gradients (e.g., first conv layer)
    assert model.conv1[0].weight.grad is not None


# ---------------------------------------------------------
# Test Loss Function
# ---------------------------------------------------------

def test_focal_loss_ignore_index():
    """Test that pixels with ignore_label (2) do not contribute to loss."""
    loss_fn = FocalLoss(ignore_index=2)

    # Batch size 1, 2 classes, 2x2 image
    logits = torch.randn(1, 2, 2, 2)

    # Target has all 2s (ignore)
    targets = torch.full((1, 2, 2), 2, dtype=torch.long)

    loss = loss_fn(logits, targets)

    # Loss should be exactly 0
    assert loss.item() == 0.0


def test_focal_loss_value():
    """Test that Focal Loss is lower for confident correct predictions."""
    loss_fn = FocalLoss(gamma=2.0)

    # Case A: Model is very confident and WRONG
    # Logits: Class 0 score low, Class 1 score high. Target is 0.
    logits_wrong = torch.tensor([[[[-5.0]], [[5.0]]]])  # Shape [1, 2, 1, 1]
    target = torch.tensor([[[0]]])

    # Case B: Model is very confident and RIGHT
    # Logits: Class 0 score high, Class 1 score low. Target is 0.
    logits_right = torch.tensor([[[[5.0]], [[-5.0]]]])

    loss_wrong = loss_fn(logits_wrong, target)
    loss_right = loss_fn(logits_right, target)

    assert loss_right < loss_wrong