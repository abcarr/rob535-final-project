#!/bin/bash
#SBATCH --job-name=multitask_weight1_10epochs
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/multitask_weight1_10epochs_%j.out
#SBATCH --error=logs/multitask_weight1_10epochs_%j.err
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

# Run training for 10 epochs WITH MULTITASK LEARNING
echo "Starting MULTITASK training with weight=1.0 at: $(date)"
echo ""
echo "=========================================="
echo "Multitask Configuration:"
echo "=========================================="
echo "use_multitask_learning: true"
echo "use_semantic: true (from WoTE_agent.yaml)"
echo "multitask_weight: 1.0"
echo "num_img_classes: 7 (0=bg, 1-6=objects)"
echo "batch_size: 16 (baseline WoTE config)"
echo "max_epochs: 10"
echo "=========================================="
echo ""
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_multitask_learning=true \
agent.config.multitask_weight=1.0 \
experiment_name=WoTE/multitask_weight1_10epochs \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
dataloader.params.num_workers=4 \
dataloader.params.prefetch_factor=2 \
+dataloader.params.persistent_workers=true \
trainer.params.max_epochs=10 \
trainer.params.check_val_every_n_epoch=1 \
split=trainval

echo "Training completed at: $(date)"

# List checkpoint files
echo "Checkpoint files created:"
ls -lh exp/WoTE/multitask_weight1_10epochs/lightning_logs/version_0/checkpoints/

# Check if training logs exist
if [ -f exp/WoTE/multitask_weight1_10epochs/lightning_logs/version_0/events.out.tfevents.* ]; then
    echo "TensorBoard logs created successfully"
fi

echo "Job finished at: $(date)"
