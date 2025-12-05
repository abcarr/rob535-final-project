#!/bin/bash
#SBATCH --job-name=eval_deformable_ablation
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_deformable_ablation_%j.out
#SBATCH --error=logs/eval_deformable_ablation_%j.err
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

# Evaluate DEFORMABLE ATTENTION model
echo "Starting evaluation of Deformable Attention ablation model at: $(date)"
echo ""
echo "=========================================="
echo "Evaluation Configuration:"
echo "=========================================="
echo "Model: Deformable Attention Only (10 epochs)"
echo "Checkpoint: final_models/deformable_ablation_epoch=10.ckpt"
echo "Split: test"
echo "Metric cache: ${NAVSIM_EXP_ROOT}/metric_cache"
echo "=========================================="
echo ""

python ./navsim/planning/script/run_pdm_score.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_deformable_attention=true \
agent.checkpoint_path=/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/final_models/deformable_ablation_epoch_10.ckpt \
experiment_name=eval/deformable_ablation \
split=test \
scene_filter=navtest \
metric_cache_path=${NAVSIM_EXP_ROOT}/metric_cache

echo ""
echo "Evaluation completed at: $(date)"
echo ""
echo "Results saved to:"
echo "  exp/eval/deformable_ablation/"
echo ""
echo "Check for PDM-Score metrics in the output above"
