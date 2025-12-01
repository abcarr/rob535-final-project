#!/bin/bash
#SBATCH --job-name=download_trainval
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=20:00:00
#SBATCH --output=logs/download_trainval_%j.out
#SBATCH --error=logs/download_trainval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abcarr@umich.edu

echo "=========================================="
echo "Downloading NAVSIM Trainval Data"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $SLURM_NODELIST"
echo ""

# Navigate to project directory
cd /scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Step 1: Removing ALL trainval data"
echo "=========================================="
echo "Removing trainval_navsim_logs..."
rm -rf trainval_navsim_logs

echo "Removing trainval_sensor_blobs..."
rm -rf trainval_sensor_blobs

echo "Removing navsim_logs/trainval symlink..."
rm -f navsim_logs/trainval

echo "Removing sensor_blobs/trainval symlink..."
rm -f sensor_blobs/trainval

echo "✓ All trainval data removed"
echo ""

echo "=========================================="
echo "Step 2: Starting fresh download"
echo "=========================================="
echo "This will download ~150GB of data"
echo "Expected time: 12-18 hours"
echo ""

# Run the download script
bash download/download_trainval.sh

echo ""
echo "=========================================="
echo "Step 3: Reorganizing directory structure"
echo "=========================================="
echo "Moving trainval data to proper locations (no symlinks)..."

# Create proper directory structure
mkdir -p navsim_logs
mkdir -p sensor_blobs

# Move trainval subdirectories to main locations
if [ -d "trainval_navsim_logs/trainval" ]; then
    mv trainval_navsim_logs/trainval navsim_logs/trainval
    rmdir trainval_navsim_logs 2>/dev/null || rm -rf trainval_navsim_logs
    echo "✓ Moved navsim_logs/trainval"
fi

if [ -d "trainval_sensor_blobs/trainval" ]; then
    mv trainval_sensor_blobs/trainval sensor_blobs/trainval
    rmdir trainval_sensor_blobs 2>/dev/null || rm -rf trainval_sensor_blobs
    echo "✓ Moved sensor_blobs/trainval"
fi

# For trainval_sensor_blobs without subdirectory (if structured differently)
if [ -d "trainval_sensor_blobs" ] && [ ! -d "sensor_blobs/trainval" ]; then
    mv trainval_sensor_blobs sensor_blobs/trainval
    echo "✓ Moved sensor_blobs/trainval (alternate structure)"
fi

echo "✓ Directory structure reorganized!"
echo ""

echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo "Job finished at: $(date)"
echo ""
echo "Your trainval data is now at:"
echo "  - navsim_logs/trainval/*.db"
echo "  - sensor_blobs/trainval/<log_name>/"
echo "=========================================="
