#!/bin/bash
#SBATCH --job-name=wote_1epoch_test
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_1epoch_test_%j.out
#SBATCH --error=logs/train_1epoch_test_%j.err

# Create logs directory
mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules and activate environment
module load python3.9-anaconda
eval "$(conda shell.bash hook)"
conda activate wote
export PATH=/home/abcarr/.conda/envs/wote/bin:$PATH

# Set environment variables
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/yingyan.li/data/navsim/maps"
export NAVSIM_EXP_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/exp"
export NAVSIM_DEVKIT_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/"
export OPENSCENE_DATA_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project"
export PYTHONNOUSERSITE=1

# Navigate to project
cd /scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project

# Verify GPU
echo "GPU Information:"
nvidia-smi

# Run training for 1 epoch
echo "Starting training at: $(date)"
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
experiment_name=WoTE/1epoch_validation \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
trainer.params.max_epochs=1 \
split=trainval

echo "Training completed at: $(date)"

# List checkpoint files
echo "Checkpoint files created:"
ls -lh exp/WoTE/1epoch_validation/lightning_logs/version_0/checkpoints/

echo "Job finished at: $(date)"
