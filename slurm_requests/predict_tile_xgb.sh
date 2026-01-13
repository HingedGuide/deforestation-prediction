#!/bin/bash
#SBATCH --job-name=predict_gabon_tile
#SBATCH --output=logs/predict_tile_%j.out
#SBATCH --error=logs/predict_tile_%j.err
#SBATCH --time=02:00:00        # Usually faster than training
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1               # Recommended for faster inference

# 1. Initialize environment
eval "$(mamba shell hook --shell bash)"
mamba activate ties_env

# 2. Go to the project root and set the PYTHONPATH
cd ~/DL_Deforestation_Prediction
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 3. Run the prediction script
# Adjust the variables below to your specific model and checkpoint
python src/deforestation_predictor/visualization/predict_tile.py \
    --data_root data/processed/input \
    --maxima_file data/processed_3d/GABON/maxima.json \
    --checkpoint checkpoints_6m_horizon/resunet3d_12m_best.pth \
    --model_type resunet3d \
    --tile_id GABON \
    --date "2024-01-01" \
    --context_months 12 \
    --output_path results/GABON_2024_01_prediction.tif