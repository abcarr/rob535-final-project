#!/bin/bash
#SBATCH --job-name=eval_1ep_ckpt
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_1epoch_checkpoint_%j.out
#SBATCH --error=logs/eval_1epoch_checkpoint_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abcarr@umich.edu

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Evaluating 1-Epoch Checkpoint"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURM_NODELIST"
echo ""

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
echo ""

echo "=========================================="
echo "Configuration:"
echo "=========================================="
echo "Checkpoint: exp/WoTE/1epoch_with_checkpoint/lightning_logs/version_36964712/checkpoints/epoch=0-step=5320.ckpt"
echo "Split: test"
echo "Scene filter: navtest"
echo "=========================================="
echo ""

# Run evaluation
echo "Starting evaluation at: $(date)"
python ./navsim/planning/script/run_pdm_score.py \
agent=WoTE_agent \
agent.checkpoint_path="exp/WoTE/1epoch_with_checkpoint/lightning_logs/version_36964712/checkpoints/epoch=0-step=5320.ckpt" \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
experiment_name=eval/WoTE/1epoch_checkpoint \
split=test \
scene_filter=navtest

echo "Evaluation completed at: $(date)"

# Check results
echo ""
echo "=========================================="
echo "Checking Results:"
echo "=========================================="

# Look for PDM score output
if [ -d "exp/eval/WoTE/1epoch_checkpoint" ]; then
    echo "✓ Evaluation results directory created"
    ls -lh exp/eval/WoTE/1epoch_checkpoint/
else
    echo "✗ No evaluation results found"
fi

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
