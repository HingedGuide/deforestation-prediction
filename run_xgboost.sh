#!/bin/bash
#SBATCH --job-name=xgb_full          # Distinct name for the full run
#SBATCH --output=logs/xgb_full_%j.out
#SBATCH --error=logs/xgb_full_%j.err
#SBATCH --time=04:00:00              # 4 hours (Data loading will be the bottleneck)
#SBATCH --mem=32G                    # 32GB RAM (Needed for 17k patches)
#SBATCH --cpus-per-task=4            # 4 Cores for background tasks
#SBATCH --gpus=1                     # 1 GPU for XGBoost acceleration

# --- SETUP ---
eval "$(mamba shell hook --shell bash)"
mamba activate ties_env

# Navigate to root and set python path
cd ~/DL_Deforestation_Prediction
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --- RUN ---
echo "Starting full training run..."
python src/deforestation_predictor/training/train_baseline_xgboost.py