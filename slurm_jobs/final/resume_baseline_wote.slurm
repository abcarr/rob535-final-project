#!/bin/bash
#SBATCH --job-name=resume_baseline_wote
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/resume_baseline_wote_%j.out
#SBATCH --error=logs/resume_baseline_wote_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abcarr@umich.edu

# Navigate to project first
cd /scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project

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
export NUPLAN_MAPS_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/maps"
export NAVSIM_EXP_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/exp"
export NAVSIM_DEVKIT_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/"
export OPENSCENE_DATA_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project"
export PYTHONNOUSERSITE=1

# Verify GPU
echo "GPU Information:"
nvidia-smi

# Resume BASELINE WOTE training (continuing to 10 epochs)
echo "Resuming BASELINE WOTE training at: $(date)"
echo ""
echo "=========================================="
echo "Resume Ablation Study: Baseline WoTE"
echo "=========================================="
echo "Checkpoint: exp/WoTE/baseline_ablation/lightning_logs/version_37339293/checkpoints/last.ckpt"
echo "use_deformable_attention: false"
echo "use_multitask_learning: false"
echo "use_convgru: false"
echo "Resuming from: Last saved checkpoint"
echo "Target: 10 total epochs"
echo "Validation: Every 1 epoch"
echo "Expected time: ~1.65-1.67 it/s"
echo "Checkpoint saving: save_top_k=-1 (all epochs) + save_last=true"
echo "=========================================="
echo ""
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.checkpoint_path=/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/exp/WoTE/baseline_ablation/lightning_logs/version_37339293/checkpoints/last.ckpt \
experiment_name=WoTE/baseline_ablation \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
dataloader.params.num_workers=8 \
dataloader.params.prefetch_factor=4 \
+dataloader.params.persistent_workers=true \
trainer.params.max_epochs=10 \
trainer.params.check_val_every_n_epoch=1 \
trainer.checkpoint.save_top_k=-1 \
trainer.checkpoint.save_last=true \
split=trainval

echo "Training completed at: $(date)"

# List checkpoint files
echo ""
echo "Checkpoint files created:"
ls -lh exp/WoTE/baseline_ablation/lightning_logs/version_37339293/checkpoints/

echo ""
echo "Training summary:"
echo "  - Configuration: Baseline WoTE (no deformable, no multitask, no ConvGRU)"
echo "  - Resumed from last checkpoint"
echo "  - Target: 10 total epochs"
echo "  - Validation: Every 1 epoch"
echo "  - Checkpoints: All epochs saved + last.ckpt"
echo "  - Average speed: ~1.65-1.67 it/s"
