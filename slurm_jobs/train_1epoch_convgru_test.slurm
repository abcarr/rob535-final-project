#!/bin/bash
#SBATCH --job-name=wote_convgru_1ep
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_convgru_1epoch_%j.out
#SBATCH --error=logs/train_convgru_1epoch_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abcarr@umich.edu

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "ConvGRU Temporal Fusion - 1 Epoch Test"
echo "=========================================="
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

# Print ConvGRU settings
echo ""
echo "=========================================="
echo "ConvGRU Configuration:"
echo "=========================================="
echo "use_convgru: true"
echo "num_history_frames: 4"
echo "slice_indices: [0, 1, 2, 3]"
echo "batch_size: 4 (small batch for quick test)"
echo "temporal_hidden_dim: 512"
echo "=========================================="
echo ""

# Run training for 1 epoch with ConvGRU temporal fusion
echo "Starting ConvGRU training at: $(date)"
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
+agent.config.use_convgru=true \
+agent.config.num_history_frames=4 \
+agent.slice_indices=[0,1,2,3] \
experiment_name=WoTE/convgru_1epoch_test \
scene_filter=navtrain \
dataloader.params.batch_size=4 \
dataloader.params.num_workers=4 \
trainer.params.max_epochs=1 \
trainer.params.check_val_every_n_epoch=1 \
split=trainval

echo "Training completed at: $(date)"

# Check results
echo ""
echo "=========================================="
echo "Verifying Results:"
echo "=========================================="

# List checkpoint files
echo "Searching for checkpoint files..."
find exp/WoTE/convgru_1epoch_test -name "*.ckpt" 2>/dev/null || echo "No checkpoints found"

# Check lightning_logs structure
echo ""
echo "Lightning logs structure:"
ls -lhR exp/WoTE/convgru_1epoch_test/lightning_logs/ 2>/dev/null || echo "No lightning_logs found"

# Check for any errors in output
echo ""
echo "Checking for GPU memory errors..."
grep -i "out of memory\|cuda\|gpu" logs/train_convgru_1epoch_$SLURM_JOB_ID.err | head -10 || echo "No memory errors found"

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
