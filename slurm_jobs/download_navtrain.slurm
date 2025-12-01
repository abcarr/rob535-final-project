#!/bin/bash
#SBATCH --job-name=download_navtrain
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/download_navtrain_%j.out
#SBATCH --error=logs/download_navtrain_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abcarr@umich.edu

echo "=========================================="
echo "Download NAVSIM Navtrain Dataset"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

cd /scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project
mkdir -p logs

# Clean up any previous partial downloads
echo "Cleaning up previous downloads..."
rm -rf openscene-v1.1 trainval_navsim_logs trainval_sensor_blobs
rm -f *.tgz

echo ""
echo "=========================================="
echo "Step 1: Download and extract metadata"
echo "=========================================="
wget https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_metadata_trainval.tgz
tar -xzf openscene_metadata_trainval.tgz
rm openscene_metadata_trainval.tgz
mv openscene-v1.1/meta_datas trainval_navsim_logs
rm -r openscene-v1.1

echo ""
echo "=========================================="
echo "Step 2: Download and extract sensor data"
echo "=========================================="
mkdir -p trainval_sensor_blobs/trainval

for split in {1..4}; do
    echo ""
    echo "Downloading navtrain_current_${split}.tgz (file $split/4)..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_current_${split}.tgz
    
    echo "Extracting navtrain_current_${split}.tgz..."
    tar -xzf navtrain_current_${split}.tgz
    rm navtrain_current_${split}.tgz
    
    echo "Moving sensor data to trainval_sensor_blobs/trainval..."
    mv -v current_split_${split}/* trainval_sensor_blobs/trainval
    rm -r current_split_${split}
    
    sensor_count=$(find trainval_sensor_blobs/trainval -type d -name "2021.*" 2>/dev/null | wc -l)
    echo "✓ Sensor directories so far: $sensor_count"
done

echo ""
echo "=========================================="
echo "Step 3: Reorganize to final structure"
echo "=========================================="
echo "Moving to final directory structure..."

# Move to final locations
mkdir -p navsim_logs sensor_blobs
mv trainval_navsim_logs/* navsim_logs/
mv trainval_sensor_blobs/* sensor_blobs/
rmdir trainval_navsim_logs trainval_sensor_blobs

echo ""
echo "=========================================="
echo "✓ Download Complete!"
echo "=========================================="
echo ""
echo "Final structure:"
echo "----------------"
db_count=$(find navsim_logs/trainval -name "*.db" 2>/dev/null | wc -l)
echo "Database files: $db_count"
sensor_count=$(find sensor_blobs/trainval -type d -name "2021.*" 2>/dev/null | wc -l)
echo "Sensor directories: $sensor_count"
echo ""
echo "Data ready for training!"
echo ""
echo "Job finished at: $(date)"
