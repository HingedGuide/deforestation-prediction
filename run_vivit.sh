#!/bin/bash
#SBATCH --job-name=vivit_train
#SBATCH --output=logs/vivit_%j.out
#SBATCH --error=logs/vivit_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1

# 1. Initialize
eval "$(mamba shell hook --shell bash)"
mamba activate ties_env

# 2. Go to project root
cd ~/DL_Deforestation_Prediction
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 3. Run ViViT
# Note: Batch size reduced to 16 to prevent CUDA Out of Memory errors.
python src/deforestation_predictor/training/train_experiment.py \
    --data_root ~/DL_Deforestation_Prediction/data/processed_3d/GABON \
    --model_type vivit \
    --context_months 6 \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --wandb_project "deforestation-prediction" \
    --wandb_run_name "ViViT_Experiment"