#!/bin/bash
#SBATCH --job-name=ablation_study_multitask
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablation_study_multitask_%j.out
#SBATCH --error=logs/ablation_study_multitask_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=%u@umich.edu

# Create logs directory
mkdir -p logs

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "User: $USER"
echo "Working directory: $(pwd)"

# Load modules and activate environment
module load python3.9-anaconda
eval "$(conda shell.bash hook)"
conda activate wote

# Set environment variables (using current directory)
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$(pwd)/maps"
export NAVSIM_EXP_ROOT="$(pwd)/exp"
export NAVSIM_DEVKIT_ROOT="$(pwd)/"
export OPENSCENE_DATA_ROOT="$(pwd)"
export PYTHONNOUSERSITE=1

# Verify GPU
echo "GPU Information:"
nvidia-smi

# Train MULTITASK ONLY from scratch (Ablation Study)
echo "Starting MULTITASK ONLY training from scratch at: $(date)"
echo ""
echo "=========================================="
echo "Ablation Study: Multi-Task Learning Only"
echo "=========================================="
echo "use_deformable_attention: false"
echo "use_multitask_learning: true"
echo "multitask_weight: 1.0"
echo "use_convgru: false"
echo "Starting from: Pretrained ResNet backbone only"
echo "Total epochs: 10"
echo "Validation: Every 2 epochs (epochs 1, 3, 5, 7, 9)"
echo "Expected time: ~7.8 hours @ 0.76 it/s (slower due to multitask)"
echo "Checkpoint saving: save_top_k=-1 (all 10 epochs) + save_last=true"
echo "=========================================="
echo ""
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_multitask_learning=true \
agent.config.multitask_weight=1.0 \
experiment_name=WoTE/multitask_only_ablation \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
dataloader.params.num_workers=8 \
dataloader.params.prefetch_factor=4 \
+dataloader.params.persistent_workers=true \
trainer.params.max_epochs=10 \
trainer.params.check_val_every_n_epoch=2 \
trainer.checkpoint.save_top_k=-1 \
trainer.checkpoint.save_last=true \
split=trainval

echo "Training completed at: $(date)"

# List checkpoint files
echo ""
echo "Checkpoint files created:"
ls -lh exp/WoTE/multitask_only_ablation/lightning_logs/version_*/checkpoints/

echo ""
echo "Training summary:"
echo "  - Configuration: Multi-Task Learning ONLY (no deformable, no ConvGRU)"
echo "  - Trained from scratch: Epochs 0-9 (10 total epochs)"
echo "  - Validation: Every 2 epochs (epochs 1, 3, 5, 7, 9)"
echo "  - Checkpoints: All 10 epochs saved + last.ckpt"
echo "  - Average speed: ~0.76 it/s"
echo "  - Total training time: ~7.8 hours"
