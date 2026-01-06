#!/bin/bash
#SBATCH --job-name=xgboost_baseline_3m
#SBATCH --output=logs/xgboost_baseline_3m_%j.out
#SBATCH --error=logs/xgboost_baseline_3m_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus=1

# 1. Initialize
eval "$(mamba shell hook --shell bash)"
mamba activate ties_env

# 2. Go to project root
cd ~/DL_Deforestation_Prediction
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 3. Run XGBoost Baseline
# Omdat dit script momenteel geen argumenten accepteert, 
# worden instellingen zoals DATA_ROOT en N_ESTIMATORS direct in de 'CONFIG' sectie 
# van train_baseline_xgboost.py beheerd.
python src/deforestation_predictor/training/train_baseline_xgboost.py