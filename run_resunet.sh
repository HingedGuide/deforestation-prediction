#!/bin/bash
#SBATCH --job-name=resunet_train
#SBATCH --output=logs/resunet_%j.out
#SBATCH --error=logs/resunet_%j.err
#SBATCH --time=12:00:00              # 12 hours
#SBATCH --cpus-per-task=4            # 4 CPU cores
#SBATCH --mem=32G                    # 32GB RAM
#SBATCH --gpus=1                     # 1 GPU

# 1. Initialize
eval "$(mamba shell hook --shell bash)"
mamba activate ties_env

# 2. Go to project root
cd ~/DL_Deforestation_Prediction
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 3. Run Training (Note: context_months=6 to match paper)
python src/deforestation_predictor/training/train_experiment.py \
    --data_root ~/DL_Deforestation_Prediction/data/processed_3d/GABON \
    --model_type resunet \
    --context_months 6 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --wandb_project "deforestation-prediction" \
    --wandb_run_name "ResUNet_Paper_Repro"