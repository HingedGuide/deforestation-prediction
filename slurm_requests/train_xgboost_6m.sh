#!/bin/bash
#SBATCH --job-name=xgboost_6m
#SBATCH --output=logs/xgboost_6%j.out
#SBATCH --error=logs/xgboost_6%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=0

# 1. Initialize
eval "$(mamba shell hook --shell bash)"
mamba activate ties_env

# 2. Go to project root
cd ~/DL_Deforestation_Prediction
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 3. Run XGBoost Baseline (6 Months)
python src/deforestation_predictor/training/train_baseline_xgboost.py \
    --data_root ~/DL_Deforestation_Prediction/data/processed_3d/GABON \
    --context_months 6