import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from deforestation_predictor.utils.logger import setup_logger

logger = setup_logger(__name__)


class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for loading 3D deforestation patches (.npz).
    Supports dynamic temporal slicing (RQ2) AND single-snapshot sampling (Laura's replication).
    """

    def __init__(self, data_root: str | Path, split: str, context_length: int | None = None, mode: str = 'sequence'):
        """
        Args:
            data_root (str | Path): Root directory containing processed data.
            split (str): 'train', 'val', or 'test'.
            context_length (int, optional): If set, limits the input to the last N months.
                                            Used for experimenting with window sizes (RQ2).
            mode (str): 'sequence' (default) - Returns a sequence of length context_length.
                        'snapshot' - Returns a single time step (random for train, last for val/test).
        """
        self.split_dir = Path(data_root) / split
        self.files = list(self.split_dir.glob("*.npz"))
        self.context_length = context_length
        self.split = split  # Neccesary to know wheter shuffling is required in snapshot mode
        self.mode = mode    # 'sequence' or 'snapshot'

        if len(self.files) == 0:
            logger.error(f"No .npz files found in {self.split_dir}")
            raise ValueError(f"No .npz files found in {self.split_dir}")

        logger.info(f"Initialized {split} dataset from {self.split_dir}")
        logger.info(f"Found {len(self.files)} samples. Mode: {self.mode}")

        if self.context_length and self.mode == 'sequence':
            logger.info(f"Sequence Mode: Temporal slicing enabled. Keeping last {self.context_length} months.")

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

            current_T = X.shape[1]

            # --- MODE 1: SNAPSHOT  ---
            if self.mode == 'snapshot':
                if self.split == 'train':
                    # Laura's method: random time step during training 
                    t_idx = np.random.randint(0, current_T)
                else:
                    # Validation/Test: last time step
                    t_idx = current_T - 1
                
                # We keep the dimension (T=1) so that models don't crash: [C, 1, H, W]
                X = X[:, t_idx:t_idx+1, :, :]

            # --- MODE 2: SEQUENCE  ---
            elif self.context_length is not None:
                if current_T >= self.context_length:
                    # Take the last 'context_length' time steps
                    X = X[:, -self.context_length:, :, :]
                else:
                    # Warn only once or handle gracefully
                    pass

            return torch.from_numpy(X), torch.from_numpy(y)

        except Exception as e:
            logger.error(f"Failed to load sample at {path}: {e}")
            raise e