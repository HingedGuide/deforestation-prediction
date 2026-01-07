#!/bin/bash
#SBATCH --job-name=resunet_12m
#SBATCH --output=logs/resunet_12m_%j.out
#SBATCH --error=logs/resunet_12m_%j.err
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

# 3. Run ConvLSTM
# Using arguments: 6 months context, 100 epochs, batch size 32
python src/deforestation_predictor/training/train_experiment.py \
    --data_root ~/DL_Deforestation_Prediction/data/processed_3d/GABON \
    --model_type resunet \
    --context_months 12 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --wandb_project "deforestation-prediction" \
    --wandb_run_name "ResUNet_12m"