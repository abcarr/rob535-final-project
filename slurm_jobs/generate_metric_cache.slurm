#!/bin/bash
#SBATCH --job-name=metric_cache
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/metric_cache_%j.out
#SBATCH --error=logs/metric_cache_%j.err

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Generating Metric Cache for Test Set"
echo "=========================================="
echo "Job started at: $(date)"

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

echo "Generating metric cache for test split..."
echo "Output path: /home/abcarr/metric_cache"
echo ""

SPLIT=test

python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py \
split=${SPLIT} \
cache.cache_path="/home/abcarr/metric_cache" \
scene_filter.frame_interval=1

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
