# Spatio-Temporal Deforestation Prediction

This repository contains a deep learning framework for predicting deforestation events using 3D spatio-temporal satellite data. The project supports various architectures (3D CNN, ResUNet, ConvLSTM, ViViT) and includes a robust preprocessing pipeline to convert raw GeoTIFF rasters into machine-learning-ready 3D cubes.

## Directory Structure

```text
src/deforestation_predictor/
├── models/
│   └── architectures.py       # DL Models: Simple3DCNN, ResUNet, ConvLSTM, ViViT
├── preprocessing/
│   ├── build_3d_dataset.py    # Main end-to-end pipeline script
│   ├── builder.py             # Cube stacking, normalization, and sampling
│   ├── mosaic_and_clip...     # Tiling and region processing
│   └── splits.py              # Time-based Train/Val/Test splitting
├── training/
│   ├── dataset.py             # PyTorch Dataset with temporal slicing (RQ2)
│   ├── loss.py                # Focal Loss implementation
│   ├── train_experiment.py    # Main DL training loop
│   └── train_baseline_xgboost.py # Tabular ML baseline
└── utils/                     # Logging and filename parsing helpers
```

## Installation
<ol>
    <li>Clone repository</li>
    <li>Install the required dependencies:</li>
</ol>

```bash
pip install -r requirements.txt
```
## Data processing
The preprocessing pipeline coverts raw `input` and `groundtruth` rasters into processed `.npz` files containing 3D tensors `[Channels, Time, Height, Width]`.

### Key steps:
<ol>
    <li>**Mosaicing:** Merges individual tiles into a specific region (e.g. Gabon).</li>
    <li>**Windowing:** Creates temporal windows based on a defined context length and gap.</li>
    <li>**Normalization:** Computes maxima from the training set to normalize the inputs.</li>
    <li>**Sampling:** Extracts balanced spatial crops (pathces) for training.</li>
</ol>

To run the pipeline:
```bash
python -m src.deforestation_predictor.preprocessing.build_3d_dataset
```

The configuration for the pipeline (e.g. region bounds, variables, dates) can be modified directly in `build_3d_dataset.py`

## Training
### Deep learning models

Use `train_experiment.py` to train the Deep Learning models. This script integrates with Weights & Biases for experiment tracking.

**Arguments:**
<ul>
    <li>`--model_type`: Choose from `3dcnn`, `resunet`, `convlstm` or `vivit`</li>
    <li>`--context_months`: Choose a number of past months to use as input.</li>
    <li>`--data_root`: Choose a path to the directory containing `train/` and `val/` `.npz` files.
</ul>

**Example Usage:**

```bash
python -m src.deforestation_predictor.training.train_experiment \
    --data_root data/processed_3d/GABON \
    --model_type resunet \
    --context_months 12 \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4
```

### XGBoost Baseline

A pixel-wise XGBoost classifier is provided as a baseline. It flattens the 3D data into tabular format and balances the classes via downsampling.

```bash
python -m src.deforestation_predictor.training.train_baseline_xgboost
```
Note: Ensure that the `DATA_ROOT` variable inside the script points to your processed data.

## Metrics and loss
<ul>
    <li>**Loss function:** Focal loss is used to handle the extreme class imbalance between forest and deforestation pixels.</li>
    <li>**Evaluation:** The primary metric is PR-AUC (Area under the Precision-Recall Curve).</li>
    <li>**Visuals:** The training script automatically logs validation predictions (Input vs Groundtruth vs Prediction) to Weights & Biases.</li>
</ul>

## Testing

Unit tests are provided for the preprocessing and training logic.

```bash
pytest tests/
```

## License
MIT Licence - Copyright (c) 2025 Ties Kuijpers
