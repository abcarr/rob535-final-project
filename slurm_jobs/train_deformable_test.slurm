#!/bin/bash
#SBATCH --job-name=deformable_attn_test
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/deformable_attn_test_%j.out
#SBATCH --error=logs/deformable_attn_test_%j.err
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

# Install deformable-attention package if not already installed
echo "Checking deformable-attention package..."
pip show deformable-attention || pip install deformable-attention
echo ""

# Run training with DEFORMABLE ATTENTION (3 epochs for quick test)
echo "Starting DEFORMABLE ATTENTION test at: $(date)"
echo ""
echo "=========================================="
echo "Deformable Attention Configuration:"
echo "=========================================="
echo "use_deformable_attention: true"
echo "Applied to: Agent decoder BEV cross-attention"
echo "BEV size: 8×8 spatial grid"
echo "Agent queries: 30"
echo "Attention type: Sparse spatial (downsample_factor=2)"
echo "batch_size: 16"
echo "max_epochs: 3 (quick test)"
echo "validation: every 2 epochs (faster)"
echo "=========================================="
echo ""
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_deformable_attention=true \
experiment_name=WoTE/deformable_attn_test \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
dataloader.params.num_workers=6 \
dataloader.params.prefetch_factor=3 \
+dataloader.params.persistent_workers=true \
trainer.params.max_epochs=1 \
trainer.params.check_val_every_n_epoch=1 \
split=trainval

echo "Training completed at: $(date)"

# List checkpoint files
echo ""
echo "Checkpoint files created:"
ls -lh exp/WoTE/deformable_attn_test/lightning_logs/version_0/checkpoints/ 2>/dev/null || echo "No checkpoints found (job may have failed)"

# Check if training logs exist
if [ -f exp/WoTE/deformable_attn_test/lightning_logs/version_0/events.out.tfevents.* ]; then
    echo "✓ TensorBoard logs created successfully"
fi

# Check for expected output
echo ""
echo "Expected in logs:"
echo "  - '✓ Deformable cross-attention enabled: Agent decoder (30 queries → 8×8 BEV)'"
echo "  - Training iteration speed (should be ~0.9 it/s on A40)"
echo "  - Validation metrics after epoch 2"
