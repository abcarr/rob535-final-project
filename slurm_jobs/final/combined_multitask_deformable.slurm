#!/bin/bash
#SBATCH --job-name=combined_multitask_deformable
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/rob535f25s001_class_root/rob535f25s001_class/%u/rob535-final-project/logs/combined_multitask_deformable_%j.out
#SBATCH --error=/scratch/rob535f25s001_class_root/rob535f25s001_class/%u/rob535-final-project/logs/combined_multitask_deformable_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=%u@umich.edu

# Detect current user
CURRENT_USER=${USER}

# Create user's logs directory
mkdir -p /scratch/rob535f25s001_class_root/rob535f25s001_class/${CURRENT_USER}/rob535-final-project/logs

# Navigate to abcarr's project directory (everyone runs from here)
cd /scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "User: $CURRENT_USER"
echo "Working directory: $(pwd)"

# Load modules and activate environment
module load python3.9-anaconda
eval "$(conda shell.bash hook)"
conda activate wote
export PATH=/home/${CURRENT_USER}/.conda/envs/wote/bin:$PATH

# Set environment variables
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/maps"
export NAVSIM_EXP_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/${CURRENT_USER}/rob535-final-project/exp"
export NAVSIM_DEVKIT_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/"
export OPENSCENE_DATA_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project"
export PYTHONNOUSERSITE=1

# Force Python to use code from current directory, not user's installed packages
export PYTHONPATH="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project:${PYTHONPATH}"
unset PYTHONUSERBASE

# Verify GPU
echo "GPU Information:"
nvidia-smi

# Train COMBINED (Multitask + Deformable) from scratch (Ablation Study)
echo "Starting COMBINED training from scratch at: $(date)"
echo ""
echo "=========================================="
echo "Ablation Study: Combined Multitask + Deformable"
echo "=========================================="
echo "use_deformable_attention: true"
echo "use_multitask_learning: true"
echo "multitask_weight: 1.0"
echo "use_convgru: false"
echo "Starting from: Pretrained ResNet backbone only"
echo "Total epochs: 8"
echo "Validation: Every 2 epochs (epochs 1, 3, 5, 7)"
echo "Expected time: ~7.8 hours @ 0.90 it/s"
echo "Checkpoint saving: save_top_k=-1 (all 8 epochs) + save_last=true"
echo "=========================================="
echo ""
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_deformable_attention=true \
agent.config.use_multitask_learning=true \
agent.config.multitask_weight=1.0 \
experiment_name=WoTE/combined_ablation \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
dataloader.params.num_workers=8 \
dataloader.params.prefetch_factor=4 \
+dataloader.params.persistent_workers=true \
trainer.params.max_epochs=8 \
trainer.params.check_val_every_n_epoch=2 \
trainer.checkpoint.save_top_k=-1 \
trainer.checkpoint.save_last=true \
split=trainval

echo "Training completed at: $(date)"

# List checkpoint files
echo ""
echo "Checkpoint files created:"
ls -lh exp/WoTE/combined_ablation/lightning_logs/version_*/checkpoints/

echo ""
echo "Training summary:"
echo "  - Configuration: COMBINED (Multi-Task + Deformable Attention)"
echo "  - Trained from scratch: Epochs 0-7 (8 total epochs)"
echo "  - Validation: Every 2 epochs (epochs 1, 3, 5, 7)"
echo "  - Checkpoints: All 8 epochs saved + last.ckpt"
echo "  - Average speed: ~0.90 it/s"
echo "  - Total training time: ~7.8 hours"
