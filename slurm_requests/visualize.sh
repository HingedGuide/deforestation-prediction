#!/bin/bash
#SBATCH --job-name=convlstm3d_6m
#SBATCH --output=logs/visualize_%j.out
#SBATCH --error=logs/visualize_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1

eval "$(mamba shell hook --shell bash)"
mamba activate ties_env

# 2. Go to project root
cd ~/DL_Deforestation_Prediction
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Example: Visualize ResUNet (12 months)
python src/deforestation_predictor/visualization/visualize_predictions.py \
    --data_root ~/DL_Deforestation_Prediction/data/processed_3d/GABON \
    --checkpoint checkpoints/ResUNet_12m_best.pth \
    --model_type resunet \
    --context_months 12 \
    --output_dir results/maps_resunet_12m \
    --num_samples 20