import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from deforestation_predictor.utils.logger import setup_logger

logger = setup_logger(__name__)

class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for On-the-Fly 3D sampling from large memory-mapped files.
    Replaces the old static patch dataset.
    """

    def __init__(self, 
                 data_root: str | Path, 
                 split: str, 
                 context_length: int = 12, 
                 mode: str = 'sequence',
                 epoch_size: int = 10000,
                 crop_size: int = 64,
                 balance_prob: float = 0.5):
        
        self.data_root = Path(data_root)
        self.split = split
        self.context_length = context_length
        self.mode = mode # 'sequence' or 'snapshot'
        self.epoch_size = epoch_size
        self.crop_size = crop_size
        self.balance_prob = balance_prob

        # 1. Load Metadata
        stats_path = self.data_root / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"stats.json not found at {self.data_root}. Did you run preprocessing?")
            
        with open(stats_path, "r") as f:
            self.stats = json.load(f)
        
        self.dates = self.stats["dates"]
        
        # 2. Determine Valid Time Indices (Hardcoded Split Logic from Preprocessing)
        # Train: < 2023-04-01 | Val: 2023-10-01 to 2024-03-01 | Test: > 2024-04-01
        valid_indices = []
        for t, d_str in enumerate(self.dates):
            if split == 'train':
                if d_str <= "2023-04-01":
                    valid_indices.append(t)
            elif split == 'val':
                if "2023-10-01" <= d_str <= "2024-03-01":
                    valid_indices.append(t)
            elif split == 'test':
                if d_str >= "2024-04-01":
                    valid_indices.append(t)
        
        # Filter indices that have enough context
        # We need data from t - context_length (exclusive) to t (inclusive)
        # So t must be >= context_length
        self.valid_time_indices = [t for t in valid_indices if t >= self.context_length]
        
        if not self.valid_time_indices:
            # Fallback for testing/debugging if splits are too tight
            logger.warning(f"No valid time indices for {split}. Using all available.")
            self.valid_time_indices = [t for t in range(len(self.dates)) if t >= self.context_length]

        # 3. Load Positive Indices for Balancing
        pos_path = self.data_root / "positive_indices.npy"
        if pos_path.exists():
            all_positives = np.load(pos_path)
            # Filter positives to only include valid time steps for this split
            mask = np.isin(all_positives[:, 0], self.valid_time_indices)
            self.split_positives = all_positives[mask]
        else:
            self.split_positives = []
        
        if len(self.split_positives) == 0 and split == 'train':
            logger.warning(f"No positive labels found in {split} split. Balancing disabled.")
            self.balance_prob = 0.0

        # 4. Open Memory Maps
        self.features_path = self.data_root / "features.npy"
        self.labels_path = self.data_root / "labels.npy"
        
        self.X_mmap = np.load(self.features_path, mmap_mode='r')
        self.y_mmap = np.load(self.labels_path, mmap_mode='r')
        
        self.C, self.T, self.H, self.W = self.X_mmap.shape

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Instellingen voor kwaliteitscontrole
        min_defo_pixels = 100  # <--- HIER KUN JE MEE SPELEN (Wil je minimaal 10, 20, 50 pixels?)
        max_attempts = 20     # Hoe vaak proberen we een 'rijke' crop te vinden?

        # 1. Select Time Step t
        use_positive = (self.split == 'train') and (np.random.rand() < self.balance_prob)
        
        t, y, x = 0, 0, 0
        found_good_crop = False

        if use_positive and len(self.split_positives) > 0:
            # --- START RETRY LOOP ---
            attempts = 0
            while attempts < max_attempts:
                # Pick from positives
                rnd_idx = np.random.randint(len(self.split_positives))
                t, center_y, center_x = self.split_positives[rnd_idx]
                
                # Center the crop
                y = center_y - self.crop_size // 2
                x = center_x - self.crop_size // 2
                
                # Jitter (Variatie toevoegen)
                y += np.random.randint(-5, 6)
                x += np.random.randint(-5, 6)

                # Boundary Checks
                y = max(0, min(y, self.H - self.crop_size))
                x = max(0, min(x, self.W - self.crop_size))

                # CHECK: Is deze crop goed genoeg?
                # We checken alleen even snel de label-map (dat is snel via mmap)
                # Let op: t-1 omdat labels op t-1 zitten in jouw logica
                y_check = self.y_mmap[t-1, y:y+self.crop_size, x:x+self.crop_size]
                
                if (y_check == 1).sum() >= min_defo_pixels:
                    found_good_crop = True
                    break # Gevonden! Uit de loop.
                
                attempts += 1
            # --- EIND RETRY LOOP ---
            
            # Als we na 20 pogingen nog niks hebben, gebruiken we gewoon de laatste (safety fallback)
            
        else:
            # Random sampling (voornamelijk bos)
            t = np.random.choice(self.valid_time_indices)
            y = np.random.randint(0, self.H - self.crop_size)
            x = np.random.randint(0, self.W - self.crop_size)
            
            # Boundary checks voor random sample
            y = max(0, min(y, self.H - self.crop_size))
            x = max(0, min(x, self.W - self.crop_size))

        # 3. Slicing (Nu halen we de data pas echt op)
        t_start = t - self.context_length
        t_end = t
        
        # Get Full Sequence
        X_crop = self.X_mmap[:, t_start:t_end, y:y+self.crop_size, x:x+self.crop_size]
        
        if self.mode == 'snapshot':
            # We keep dimensions [C, 1, H, W]
            X_crop = X_crop[:, -1:, :, :]

        y_crop = self.y_mmap[t-1, y:y+self.crop_size, x:x+self.crop_size]

        # 4. Convert to Tensor
        X_tensor = torch.from_numpy(X_crop.copy()).float()
        y_tensor = torch.from_numpy(y_crop.copy()).long()

        return X_tensor, y_tensor