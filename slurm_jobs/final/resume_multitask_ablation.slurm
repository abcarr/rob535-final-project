#!/bin/bash
#SBATCH --job-name=resume_multitask_ablation
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/resume_multitask_ablation_%j.out
#SBATCH --error=logs/resume_multitask_ablation_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=%u@umich.edu

# Navigate to user's rob535 project directory in scratch
cd /scratch/rob535f25s001_class_root/rob535f25s001_class/${USER}/rob535-final-project

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
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export PYTHONNOUSERSITE=1

# Verify GPU
echo "GPU Information:"
nvidia-smi

# Resume MULTITASK ONLY training (continuing to 10 epochs)
echo "Resuming MULTITASK ONLY training at: $(date)"
echo ""
echo "=========================================="
echo "Resume Ablation Study: Multi-Task Learning Only"
echo "=========================================="
echo "Checkpoint: /scratch/rob535f25s001_class_root/rob535f25s001_class/vmarri/rob535-final-project/exp/WoTE/baseline_ablation/lightning_logs/version_37360260/checkpoints/last.ckpt"
echo "use_deformable_attention: false"
echo "use_multitask_learning: true"
echo "multitask_weight: 1.0"
echo "use_convgru: false"
echo "Resuming from: Last saved checkpoint"
echo "Target: 10 total epochs"
echo "Validation: Every 2 epochs"
echo "Expected time: ~0.76 it/s (slower due to multitask)"
echo "Checkpoint saving: save_top_k=-1 (all epochs) + save_last=true"
echo "=========================================="
echo ""
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_multitask_learning=true \
agent.config.multitask_weight=1.0 \
agent.checkpoint_path=/scratch/rob535f25s001_class_root/rob535f25s001_class/vmarri/rob535-final-project/exp/WoTE/baseline_ablation/lightning_logs/version_37360260/checkpoints/last.ckpt \
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
echo "  - Resumed from last checkpoint"
echo "  - Target: 10 total epochs"
echo "  - Validation: Every 2 epochs"
echo "  - Checkpoints: All epochs saved + last.ckpt"
echo "  - Average speed: ~0.76 it/s"
