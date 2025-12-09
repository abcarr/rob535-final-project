#!/bin/bash
#SBATCH --job-name=multitask_deformable_test
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/multitask_deformable_test_%j.out
#SBATCH --error=logs/multitask_deformable_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abcarr@umich.edu

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

# Navigate to project
cd /scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project

# Verify GPU
echo "GPU Information:"
nvidia-smi

# Run training with BOTH MULTITASK + DEFORMABLE ATTENTION
echo "Starting MULTITASK + DEFORMABLE ATTENTION test at: $(date)"
echo ""
echo "=========================================="
echo "Combined Configuration:"
echo "=========================================="
echo "use_multitask_learning: true"
echo "  - Image segmentation (7 classes)"
echo "  - Instance offsets"
echo "  - Inverse depth"
echo "  - Uncertainty weighting (Kendall et al.)"
echo "multitask_weight: 1.0"
echo ""
echo "use_deformable_attention: true"
echo "  - Agent decoder BEV refinement"
echo "  - 8×8 spatial attention"
echo "  - Learned offset sampling"
echo ""
echo "Training: 1 epoch (quick smoke test)"
echo "Validation: after epoch 1"
echo "DataLoader: batch_size=16, num_workers=8, persistent"
echo "=========================================="
echo ""
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_multitask_learning=true \
agent.config.multitask_weight=1.0 \
agent.config.use_deformable_attention=true \
experiment_name=WoTE/multitask_deformable_test \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
dataloader.params.num_workers=8 \
dataloader.params.prefetch_factor=4 \
+dataloader.params.persistent_workers=true \
trainer.params.max_epochs=1 \
trainer.params.check_val_every_n_epoch=1 \
split=trainval

echo "Training completed at: $(date)"

# List checkpoint files
echo ""
echo "Checkpoint files created:"
ls -lh exp/WoTE/multitask_deformable_test/lightning_logs/version_0/checkpoints/ 2>/dev/null || echo "No checkpoints found"

# Check for expected outputs
echo ""
echo "Expected in logs:"
echo "  - '✓ Deformable cross-attention enabled: Agent decoder (30 queries → 8×8 BEV)'"
echo "  - Multitask loss components: seg, inst, depth with uncertainty weights"
echo "  - Training speed: ~0.85-0.90 it/s (slightly slower due to both features)"
echo "  - Memory usage: ~40-45GB (both features active)"
