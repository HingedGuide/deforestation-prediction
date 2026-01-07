#!/bin/bash
#SBATCH --job-name=preprocessing_6m
#SBATCH --output=logs/preprocessing_6m_%j.out
#SBATCH --error=logs/preprocessing_6m_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=0

eval "$(mamba shell hook --shell bash)"
mamba activate /opt/mamba/envs/python312

# 2. Go to project root
cd ~/DL_Deforestation_Prediction
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 3. Run preprocessing for 6 months context
python src/deforestation_predictor/preprocessing/build_3d_dataset.py