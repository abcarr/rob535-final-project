#!/bin/bash
#SBATCH --job-name=download_tarballs
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/download_tarballs_%j.out
#SBATCH --error=logs/download_tarballs_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=abcarr@umich.edu

echo "=========================================="
echo "Download NAVSIM Navtrain Tarballs (NO EXTRACTION)"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

cd /scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project
mkdir -p logs
mkdir -p navtrain_tarballs

echo "Current directory: $(pwd)"
echo ""
echo "This will download ~445GB of compressed tarballs"
echo "Storage location: navtrain_tarballs/"
echo "NO extraction - files stay compressed"
echo "Estimated time: 4-6 hours"
echo ""

# Navigate to tarball storage directory
cd navtrain_tarballs

echo "=========================================="
echo "Downloading Current Frame Tarballs (4 files)"
echo "=========================================="
echo ""

for i in {1..4}; do
    echo "----------------------------------------"
    echo "Downloading navtrain_current_${i}.tgz (${i}/4)"
    echo "----------------------------------------"
    
    if [ -f "navtrain_current_${i}.tgz" ]; then
        echo "✓ navtrain_current_${i}.tgz already exists, skipping..."
    else
        wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_current_${i}.tgz
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to download navtrain_current_${i}.tgz"
            exit 1
        fi
        
        # Verify download
        size=$(du -h navtrain_current_${i}.tgz | cut -f1)
        echo "✓ Downloaded: $size"
    fi
    echo ""
done

echo "=========================================="
echo "Downloading Historical Frame Tarballs (4 files)"
echo "=========================================="
echo ""

for i in {1..4}; do
    echo "----------------------------------------"
    echo "Downloading navtrain_history_${i}.tgz (${i}/4)"
    echo "----------------------------------------"
    
    if [ -f "navtrain_history_${i}.tgz" ]; then
        echo "✓ navtrain_history_${i}.tgz already exists, skipping..."
    else
        wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/navsim/navtrain_history_${i}.tgz
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to download navtrain_history_${i}.tgz"
            exit 1
        fi
        
        # Verify download
        size=$(du -h navtrain_history_${i}.tgz | cut -f1)
        echo "✓ Downloaded: $size"
    fi
    echo ""
done

echo "=========================================="
echo "✓ All Tarballs Downloaded Successfully!"
echo "=========================================="
echo ""

# Show final statistics
echo "Final Summary:"
echo "----------------"
echo "Current frame tarballs:"
ls -lh navtrain_current_*.tgz 2>/dev/null | awk '{print "  - " $9 ": " $5}'

echo ""
echo "Historical frame tarballs:"
ls -lh navtrain_history_*.tgz 2>/dev/null | awk '{print "  - " $9 ": " $5}'

echo ""
echo "Total storage used:"
du -sh . | awk '{print "  " $1}'

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "Tarballs are stored in: $(pwd)"
echo ""
echo "To extract specific splits later:"
echo "  1. Clean up current sensor_blobs/"
echo "  2. Extract only the splits you need"
echo "  3. Example: tar -xzf navtrain_current_1.tgz -C ../sensor_blobs/trainval/"
echo "  4. Example: tar -xzf navtrain_history_1.tgz -C ../sensor_blobs/trainval/"
echo ""
echo "File counts per split (approximate):"
echo "  - Current (1 split):  ~90k files  (frame 3 only)"
echo "  - History (1 split):  ~270k files (frames 0,1,2)"
echo "  - Both (1 split):     ~360k files (frames 0,1,2,3)"
echo ""
echo "With 1M file quota, you can fit ~2-3 complete splits with history"
echo ""
echo "Job finished at: $(date)"
