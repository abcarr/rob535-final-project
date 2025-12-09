#!/bin/bash
#SBATCH --job-name=deformable_resume
#SBATCH --account=rob535f25s001_class
#SBATCH --partition=spgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/deformable_resume_%j.out
#SBATCH --error=logs/deformable_resume_%j.err
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

# Resume training with DEFORMABLE ATTENTION (8 additional epochs)
echo "Resuming DEFORMABLE ATTENTION training at: $(date)"
echo ""
echo "=========================================="
echo "Resume Configuration:"
echo "=========================================="
echo "Checkpoint: exp/WoTE/deformable_attn_test/lightning_logs/version_37183486/checkpoints/epoch=0-step=5320.ckpt"
echo "use_deformable_attention: true"
echo "Starting from: Epoch 0 (completed)"
echo "Target: 9 total epochs (8 additional epochs)"
echo "Expected time: ~7 hours for 8 additional epochs"
echo "=========================================="
echo ""
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
agent.config.use_deformable_attention=true \
agent.checkpoint_path=exp/WoTE/deformable_attn_test/lightning_logs/version_37183486/checkpoints/epoch=0-step=5320.ckpt \
experiment_name=WoTE/deformable_attn_test \
scene_filter=navtrain \
dataloader.params.batch_size=16 \
dataloader.params.num_workers=8 \
dataloader.params.prefetch_factor=4 \
+dataloader.params.persistent_workers=true \
trainer.params.max_epochs=9 \
trainer.params.check_val_every_n_epoch=1 \
split=trainval

echo "Training completed at: $(date)"

# List checkpoint files
echo ""
echo "Checkpoint files created:"
ls -lh exp/WoTE/deformable_attn_test/lightning_logs/version_37183486/checkpoints/

echo ""
echo "Training summary:"
echo "  - Resumed from epoch 0 (completed)"
echo "  - Trained: Epochs 1-8 (8 additional epochs)"
echo "  - Total: 9 epochs (0-8)"
echo "  - Average speed: ~1.7 it/s"
echo "  - Total time for additional epochs: ~7 hours"
