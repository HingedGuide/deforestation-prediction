# Deforestation Prediction Training Package

This package contains the complete training pipeline for the deforestation prediction models. It includes data loading, loss function definitions, and training scripts for both Deep Learning (DL) architectures and the XGBoost baseline.

## Package Structure

| File | Description |
| :--- | :--- |
| **`dataset.py`** | Defines the PyTorch `Dataset` for loading processed 3D patches `.npz`. Handles dynamic temporal slicing for Research Question 2 (RQ2). |
| **`loss.py`** | Implements `FocalLoss` to handle the heavy class imbalance between forest (majority) and deforestation (minority). |
| **`train_experiment.py`** | The main entry point for training DL models (e.g., `convLSTM`, `resunet3D`). Handles the training loop, validation, and checkpointing. |
| **`train_baseline_xgboost.py`** | A standalone script to train and evaluate the XGBoost baseline on flattened pixel data. |

---

## Key Components

### 1. Data Loading (`dataset.py`)
The `DeforestationDataset` class is responsible for loading the compressed `.npz` files created by the preprocessing pipeline.

* **Input**: Directory containing `.npz` files (e.g., `data/processed_3d/GABON/train`).
* **Output**: A tuple `(X, y)` where:
    * `X` is the input tensor of shape `[Channels, Time, Height, Width]`.
    * `y` is the target mask of shape `[Height, Width]`.
* **Temporal Slicing (RQ2)**: The class accepts a `context_length` argument. If provided, it slices the time dimension to keep only the *last N months* of data. [cite_start]This allows you to experiment with how much historical context is needed for accurate predictions[cite: 55].

### 2. Loss Function (`loss.py`)
[cite_start]We use **Weighted Focal Loss** instead of standard Cross Entropy because deforestation is a rare event (class imbalance)[cite: 31].

* [cite_start]**Ignore Label**: The loss function automatically masks out pixels with value `2` (e.g., non-forest areas, water, or outside the study region) so they do not influence the gradient[cite: 31].
* **Alpha/Gamma**: Parameters are tuned to focus the model's attention on hard-to-classify, positive examples (deforestation events).

---

## Usage

### A. Deep Learning Experiment
Use `train_experiment.py` to train 3D CNNs or ResUNets. This script supports command-line arguments to switch architectures and context lengths easily.

**Basic Command:**
```bash
python -m deforestation_predictor.training.train_experiment \
    --data_root data/processed_3d/GABON \
    --model_type 3dcnn \
    --context_months 3 \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4
```

### Arguments
* **`--data_root`**: Path to the directory containing `train/` and `val/` subfolders.
* **`--model_type`**: The architecture to use.
    * Options: `resunet`, `resunet3d`, `convlstm3d` and `vivit`
* **`--context_months`**: Number of past months to use as input.
    * Default is `12`.
    * Reducing this (e.g., to `3`) tests the model's reliance on long-term history.
* **`--save_dir`**: Where to save the `best_model.pth`.

### Outputs
* Logs are written to `training_experiment.log`.
* The best model (highest Val F0.5) is saved to `checkpoints/best_model.pth`.

### B. XGBoost Baseline

Use `train_baseline_xgboost.py` to train the tabular baseline. This script flattens the 3D spatio-temporal data into a 2D matrix (Pixels × Features).

#### Steps performed by the script:

* Flattening: Converts [Channels, Time, Height, Width] → [N_pixels, N_features].
* Balancing: Downsamples the "no-deforestation" class in the training set to save memory and improve training speed (default sample_rate=0.1).
* Training: Fits an XGBoost classifier.
* Threshold Optimization: Finds the optimal probability threshold that maximizes the F0.5 Score on the validation set (prioritizing precision).
* Evaluation: Reports Precision, Recall, and F0.5 on the test set.

**Command:**


```bash
python -m deforestation_predictor.training.train_baseline_xgboost
    --data_root data/processed_3d/GABON \
    --context_months 12
```

### Arguments
* **`--data_root`**: Path to the directory containing `train/` and `val/` subfolders.
* **`--context_months`**: Number of past months to use as input.

