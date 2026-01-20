# Visualization and Inference \
This directory contains scripts that are used for visualizeing the model predictions. It can be used for small test samples, but also for full tiles for both the Deep Learning architectures and the XGBoost baseline.

# Script overview

| File                      | Description                                                                                      |
|---------------------------|--------------------------------------------------------------------------------------------------|
| `predict_tile.py`         | Performs inference on a full GeoTIFF tile using a trained DL model via a sliding window approach |
| `predict_tile_xgboost.py` | Performs inference on a full GeoTIFF tile using a trained XGBoost model                          |
| `visualize_prediction.py` | Generates comparison plots for a batch of test samples to inspect the model performance          |


# Key components
**1. Full-Tile Inference (`predict_tile.py` & `predict_tile_xgboost.py`)** \
These scripts load raw raster data, build normalized input cubes and generate a continuous probability map (`.tif`) for a specific study area (e.g. Gabon) on a chosen date.
* Sliding window: For DL models, a sliding-window method with adjustable overlap is used to prevent memory issues when processing large tiles.
* Chunked processing: The XGBoost script processes data in chunks (e.g. 1024x1024 pixels) since the model operates on a pixel-by-pixel basis and does not require spatial context within its architecture
* Geospatial export: The results are saved as GeoTIFFs so that the original coordinates and metadata are saved from the input rasters

**2. Model Comparison (`visualize_predictions.py`)** \
This tool is used for quick qualitative analysis after training
* It displays the first channel of the last time step as a reference image
* It visualizes Ground Truth masks (including ignore labels) alongside predicted binary masks


