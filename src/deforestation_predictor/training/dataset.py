import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from deforestation_predictor.utils.logger import setup_logger

logger = setup_logger(__name__)


class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for loading 3D deforestation patches (.npz).
    Supports dynamic temporal slicing for RQ2 (Temporal Window Analysis).
    """

    def __init__(self, data_root: str | Path, split: str, context_length: int | None = None):
        """
        Args:
            data_root (str | Path): Root directory containing processed data.
            split (str): 'train', 'val', or 'test'.
            context_length (int, optional): If set, limits the input to the last N months.
                                            Used for experimenting with window sizes (RQ2).
        """
        self.split_dir = Path(data_root) / split
        self.files = list(self.split_dir.glob("*.npz"))
        self.context_length = context_length

        if len(self.files) == 0:
            logger.error(f"No .npz files found in {self.split_dir}")
            raise ValueError(f"No .npz files found in {self.split_dir}")

        logger.info(f"Initialized {split} dataset from {self.split_dir}")
        logger.info(f"Found {len(self.files)} samples.")

        if self.context_length:
            logger.info(f"RQ2 Mode: Temporal slicing enabled. Keeping last {self.context_length} months.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a single sample.

        Returns:
            X (Tensor): Shape [Channels, Time, Height, Width]
            y (Tensor): Shape [Height, Width]
        """
        path = self.files[idx]

        try:
            with np.load(path) as data:
                # X shape: [C, T, H, W] (Channels, Time, Height, Width)
                X = data['X'].astype(np.float32)
                y = data['y'].astype(np.longlong)  # Mask: 0, 1, 2(ignore)

            # --- RQ2: Temporal Window Slicing ---
            # If we only want the last N months (e.g., 3 months), we slice the time dimension.
            if self.context_length is not None:
                current_T = X.shape[1]
                if current_T >= self.context_length:
                    # Take the last 'context_length' time steps
                    X = X[:, -self.context_length:, :, :]
                else:
                    # Warn only once to prevent spamming logs, or handle gracefully
                    # Here we just return what we have, but theoretically this shouldn't happen
                    # if the preprocessing was done correctly.
                    pass

            return torch.from_numpy(X), torch.from_numpy(y)

        except Exception as e:
            logger.error(f"Failed to load sample at {path}: {e}")
            raise e