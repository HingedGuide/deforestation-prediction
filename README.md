# Spatio-Temporal Deforestation Prediction
![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Status](https://img.shields.io/badge/status-Master_Thesis-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)


This repository contains a deep learning framework for predicting deforestation events using 3D spatio-temporal satellite data. The project supports various architectures (ResUNet, ResUNet3D, ConvLSTM, ViViT) and includes a robust preprocessing pipeline to convert raw GeoTIFF rasters from the [WWF Forest Foresight](https://forestforesight.atlassian.net/wiki/spaces/EWS/overview?homepageId=32961) project into machine-learning-ready 3D cubes. This repository was created as part of my Master's Thesis.

## Results

### Model Performance
To evaluate the predictive performance of the models, the F0.5 score was used as the primary metric. This metric gives more weight to Precision than Recall, which is crucial for early warning systems where false positives need to be minimized. 

The evaluation showed that convolutional models significantly outperformed the attention-based Video Vision Transformer (ViViT). Interestingly, explicitly modeling the spatio-temporal dependencies in 3D did not result in a universal improvement. The baseline 2D ResUNet, which aggregates the temporal context into a single feature map, achieved the highest overall performance.

| Model | Architecture Type | Highest F0.5 Score |
|-------|-------------------|--------------------|
| **2D ResUNet** | Convolutional (Aggregated Temporal) | **0.6526** |
| **3D ResUNet** | Convolutional (Spatio-Temporal) | Comparable to 2D |
| **ConvLSTM3D** | Convolutional (Spatio-Temporal) | Comparable to 2D |
| **ViViT** | Attention-based (Spatio-Temporal) | Lowest Performance |

*(Note: The exact F0.5 scores vary depending on the prediction horizon and temporal context length, but the 2D ResUNet yielded the highest overall average.)*

### Key Findings
1. **Temporal Context:** A short historical context of 3 months was sufficient to capture the relevant dynamics of deforestation progression. Adding 6 or 12 months of historical data did not significantly improve the predictive value and sometimes introduced noise.
2. **Prediction Horizon:** The models performed significantly better over longer prediction horizons compared to very short-term (1-month) predictions.
3. **Generalization:** While the 3D ResUNet showed significant improvement in detecting new deforestation locations in Gabon, this performance did not perfectly generalize to other tested regions (Venezuela and Papua New Guinea) on a regional dataset.

### Visual Predictions
*(Add a side-by-side image here of a satellite image, the ground truth deforestation, and your 2D ResUNet prediction. For example: `![Prediction vs Ground Truth](assets/prediction_comparison.png)`)*

## Directory Structure

```text
src/deforestation_predictor/
├── evaluation/                    # Contains scripts that are used to evaluate the trained models
│   ├── eval_new_frontiers.py
│   ├── eval_test_set.py
│   ├── evaluate_distance.py
│   ├── evaluate_montly.py
│   ├── friedman_test.py
│   ├── nemeny_test.py
│   ├── new_frontiers_montly.py
│   └── patch_size_analysis
├── models/
│   └── architectures.py           # DL Models: Simple3DCNN, ResUNet, ConvLSTM, ViViT
├── preprocessing/
│   ├── create_country_masks.py    # split data into train, val and test and apply normalization 
│   └── organize_data_normalize.py # prepare data from training loop
├── training/
│   ├── loss.py                    # Combined Loss implementation
│   └── train_experiment.py        # Main DL training loop 
├── visualization/
│   ├── predict_tile.py            # DL Inference on a full tile
│   └── visualize_predictions.py   # Visualization of samples
└── utils/                         # Logging and filename parsing helpers
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HingedGuide/deforestation-prediction
   cd deforestation-prediction

2. Install the required dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv/Scripts/activate
   pip install -r requirements.txt
   ```
## Data processing
The preprocessing pipeline coverts raw `input` and `groundtruth` rasters into processed `.npy` files containing 3D tensors `[Channels, Time, Height, Width]`.

**Arguments**
- `--tiles`: Choose which tiles should be processed.
- `--dump_path`: Choose the directory where the processed data (numpy files) will be saved.
- `--data_input`: Choose the folder path containing the input variable files.
- `--data_gt`: Choose the folder path containing the ground-truth data.
- `--dynamic_var`: Choose the dynamic variables and their normalization values to be used.
- `--auto_gen_dynamic`: Choose which dynamic variables should be auto-generated (e.g., month).
- `--yearly_var`: Choose the yearly (semi-dynamic) variables to include.
- `--static_names`: Choose the static variables and their normalization values.
- `--ref_name`: Choose the name of the reference variable (e.g., groundtruth).
- `--mask_name`: Choose which variable is used to draw the mask for the region.
- `--normalize`: Choose whether to normalize the data based on maximum values provided by WWF (True/False).
- `--startdate_train`: Choose the start date for the training data.
- `--enddate_train`: Choose the end date for the training data.
- `--startdate_val`: Choose the start date for the validation data.
- `--enddate_val`: Choose the end date for the validation data.
- `--startdate_test`: Choose the start date for the test data.
- `--enddate_test`: Choose the end date for the test data.


**Example Usage**

```bash
python -m src.deforestation_predictor.preprocessing.organize_data_normalize \
  --tiles "00N_010E,10N_010E" \
  --dump_path "./data_npy_normby_wwf" \
  --data_input "C:/PostDoc/deforestation_project/data/tiles_march_2025/input" \
  --data_gt "C:/PostDoc/deforestation_project/data/tiles_march_2025/gt" \
  --dynamic_var "[['lastmonth','1600'],['precipitation','240'],['temperature','3000'],['timesinceloss','10000'],['totallossalerts','1600']]" \
  --auto_gen_dynamic "[['month','12'],['sinmonth','0']]" \
  --yearly_var "[['closenesstoroads', '255'],['losslastyear','256']]" \
  --static_names "[['closenesstowaterways','255'],['elevation','8849'],['historicloss','256'],['initialforestcover','10000'],['populationcurrent','20000000'],['populationincrease','20000000'],['slope','4000'],['wetlands','0'],['peatland','15'],['croplandcapacity100p','255'],['croplandcapacitybelow50p','255'],['croplandcapacityover50p','255'],['landpercentage','254'],['forestheight', '50'],['wdpa','0'],['catexcap','1000'],['xy', '0']]" \
  --ref_name "groundtruth6m" \
  --mask_name "slope" \
  --normalize True \
  --startdate_train "2021-01-01" \
  --enddate_train "2022-12-01" \
  --startdate_val "2023-06-01" \
  --enddate_val "2023-12-01" \
  --startdate_test "2024-01-01" \
  --enddate_test "2024-06-01"
```

## Training
### Deep learning models

Use `train_experiment.py` to train the Deep Learning models.

**Arguments:**
- `--image_path`: Choose where the processed input files are stored
- `--tiles`: Choose which 10x10 degree tiles should be processed.
- `--country`: Choose which country is being trained upon.
- `--save_dir`: Choose where the model checkpoint will be saved.
- `--model_type`: Choose from `3dcnn`, `resunet`, `convlstm` or `vivit`
- `--mode`: Choose between `sequence` or `snapshot`. Sequence is default for 3D, and snapshot is for 2D
- `--context_months`: Choose a number of past months to use as input.
- `--epochs`: Choose for how many epochs the model will run.
- `--batch_size`: Choose the batch size of the model.
- `--lr`: Choose the learning rate of the model.
- `--samples`: Choose how many samples the model will take from the input data.
- `--balance`: Choose the balance between balanced training (1) or imbalanced training (0).
- `--patience`: Choose after how many epochs early stopping will trigger.

**Example Usage:**

```bash
python -m src.deforestation_predictor.training.train_experiment \
  --image_path "./preprocessing/output" \
  --tiles "00N_000E" \
  --country "Laos" \
  --save_dir "checkpoints" \
  --model_type "resunet" \
  --mode "snapshot" \
  --context_months 12 \
  --epochs 50 \
  --batch_size 8 \
  --lr 0.0001 \
  --samples 50000 \
  --balance 1 \
  --patience 20
```

## Metrics and loss

- **Loss function:** Combined loss is used to handle the extreme class imbalance between forest and deforestation pixels.
- **Evaluation:** The primary metric is F0.5

Use `predict_tile.py` for predicting a full tile with a trained DL model.

## Testing

Unit tests are provided for the preprocessing and training logic.

```bash
pytest tests/
```

## License
MIT Licence - Copyright (c) 2025 Ties Kuijpers
