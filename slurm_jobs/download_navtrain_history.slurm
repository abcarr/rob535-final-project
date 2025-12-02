#!/bin/bash
#SBATCH --job-name=download_history
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/download_history_%j.out
#SBATCH --error=logs/download_history_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abcarr@umich.edu

echo "=========================================="
echo "Download NAVSIM Navtrain Historical Data"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

cd /scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project
mkdir -p logs

echo "Current directory: $(pwd)"
echo ""
echo "This will download ~145GB of historical sensor data (frames 0, 1, 2)"
echo "Estimated time: 3-4 hours"
echo ""

# Navigate to sensor_blobs/trainval directory
cd sensor_blobs/trainval

echo "=========================================="
echo "Downloading Historical Sensor Data"
echo "=========================================="
echo "Working directory: $(pwd)"
echo ""

for split in {1..4}; do
    echo ""
    echo "----------------------------------------"
    echo "Processing navtrain_history_${split}.tgz (file $split/4)"
    echo "----------------------------------------"
    
    echo "Downloading navtrain_history_${split}.tgz..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_history_${split}.tgz
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download navtrain_history_${split}.tgz"
        exit 1
    fi
    
    echo "Extracting navtrain_history_${split}.tgz..."
    tar -xzf navtrain_history_${split}.tgz
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to extract navtrain_history_${split}.tgz"
        exit 1
    fi
    
    echo "Removing tarball to save space..."
    rm navtrain_history_${split}.tgz
    
    echo "Moving history data to current directory..."
    mv history_split_${split}/* .
    rmdir history_split_${split}
    
    # Count total sensor directories
    sensor_count=$(find . -maxdepth 1 -type d -name "2021.*" 2>/dev/null | wc -l)
    echo "✓ Total sensor directories: $sensor_count"
    
    # Check one directory to see how many frames it has
    sample_dir=$(find . -maxdepth 1 -type d -name "2021.*" 2>/dev/null | head -1)
    if [ -n "$sample_dir" ]; then
        cam_count=$(ls "$sample_dir/CAM_F0/" 2>/dev/null | wc -l)
        echo "✓ Sample directory has $cam_count camera files"
    fi
done

echo ""
echo "=========================================="
echo "✓ Historical Data Download Complete!"
echo "=========================================="
echo ""

# Final verification
echo "Final Statistics:"
echo "----------------"
sensor_count=$(find . -maxdepth 1 -type d -name "2021.*" 2>/dev/null | wc -l)
echo "Total sensor directories: $sensor_count"

# Check a few directories to verify historical frames exist
echo ""
echo "Verifying historical frames in sample directories:"
for sample_dir in $(find . -maxdepth 1 -type d -name "2021.*" 2>/dev/null | head -3); do
    cam_count=$(ls "$sample_dir/CAM_F0/" 2>/dev/null | wc -l)
    echo "  - $(basename $sample_dir): $cam_count camera files"
done

echo ""
echo "Historical sensor data ready for ConvGRU training!"
echo ""
echo "You can now train with:"
echo "  +agent.slice_indices=[0,1,2,3]"
echo "  +agent.config.use_convgru=true"
echo "  +agent.config.num_history_frames=4"
echo ""
echo "Job finished at: $(date)"
