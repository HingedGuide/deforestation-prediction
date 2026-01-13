#!/bin/bash
#SBATCH --job-name=deforestation_experiment_full
#SBATCH --output=logs/slurm_startup_%A_%a.out  # Captures only startup errors
#SBATCH --error=logs/slurm_startup_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=0-11%3

# 1. Define the configurations
configs=(
    "resunet 3"
    "resunet 6"
    "resunet 12"
    "resunet3d 3"
    "resunet3d 6"
    "resunet3d 12"
    "convlstm3d 3"
    "convlstm3d 6"
    "convlstm3d 12"
    "vivit 3"
    "vivit 6"
    "vivit 12"
)

# 2. Determine which model to run
current_config=${configs[$SLURM_ARRAY_TASK_ID]}
read -r MODEL_TYPE CONTEXT_MONTHS <<< "$current_config"

# We now redirect all output (stdout and stderr) to desired filename.
# Format: logs/model_months.out
LOG_FILE="logs/${MODEL_TYPE}_${CONTEXT_MONTHS}"

echo "Output will be written to: ${LOG_FILE}.out from now on"

# Redirect stdout (1) and stderr (2) to the new files
exec > "${LOG_FILE}.out" 2> "${LOG_FILE}.err"

# From here on, everything goes into specific log file
echo "Starting task ${SLURM_ARRAY_TASK_ID}: Model=${MODEL_TYPE}, Context=${CONTEXT_MONTHS}m"
date

# 3. Initialize environment (only now, so output also goes to the correct log)
eval "$(mamba shell hook --shell bash)"
mamba activate ties_env

# 4. Go to project root
cd ~/DL_Deforestation_Prediction
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 5. Run the training script
python src/deforestation_predictor/training/train_experiment.py \
    --data_root ~/DL_Deforestation_Prediction/data/processed_3d/GABON \
    --model_type ${MODEL_TYPE} \
    --context_months ${CONTEXT_MONTHS} \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --wandb_project "deforestation-prediction" \
    --wandb_run_name "${MODEL_TYPE}_${CONTEXT_MONTHS}m"