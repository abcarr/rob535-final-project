# WoTE Setup Summary - All Changes Made

This document summarizes all the changes made to get the WoTE project working on UMich Great Lakes HPC. Use this as a reference to set up your own repo.

---

## Table of Contents
1. [Overview](#overview)
2. [Files Modified](#files-modified)
3. [Files Created](#files-created)
4. [Step-by-Step Setup Process](#step-by-step-setup-process)
5. [Key Configuration Changes](#key-configuration-changes)
6. [Downloads Required](#downloads-required)
7. [Common Issues Fixed](#common-issues-fixed)

---

## Overview

**Goal:** Get WoTE training working on Great Lakes HPC starting from the original repo

**Key Challenges Solved:**
- Hardcoded paths throughout the codebase
- Python environment activation issues
- Data structure mismatches
- Missing files and configurations
- SLURM job setup

**Total Time:** ~6-8 hours of setup before first successful training run

---

## Files Modified

### 1. `environment.yml`
**Change:** Python version from 3.8 → 3.9

**Reason:** `tensorboard==2.16.2` requires Python ≥3.9

```yaml
# Before
dependencies:
  - python=3.8

# After
dependencies:
  - python=3.9
```

### 2. `navsim/agents/WoTE/configs/default.py`
**Changes:** Updated all hardcoded paths to Great Lakes paths

**Lines modified:**
- Line 16: `resnet34_path`
- Line 136: `sim_reward_dict_path`
- Line 137: `cluster_file_path`

```python
# Before
resnet34_path = '/home/yingyan.li/repo/WoTE/ckpts/resnet34.pth'
sim_reward_dict_path: str = f'/home/yingyan.li/repo/WoTE/dataset/extra_data/planning_vb/formatted_pdm_score_{num_traj_anchor}.npy'
cluster_file_path = f'/home/yingyan.li/repo/WoTE/dataset/extra_data/planning_vb/trajectory_anchors_{num_traj_anchor}.npy'

# After (replace with YOUR scratch path)
resnet34_path = '/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/ckpts/resnet34.pth'
sim_reward_dict_path: str = f'/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/extra_data/planning_vb/formatted_pdm_score_{num_traj_anchor}.npy'
cluster_file_path = f'/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/extra_data/planning_vb/trajectory_anchors_{num_traj_anchor}.npy'
```

### 3. `scripts/training/run_wote.sh`
**Changes:** Updated all environment variables and checkpoint path

```bash
# Before
export PYTHONPATH=/home/yingyan.li/repo/WoTE/
export NUPLAN_MAPS_ROOT="/home/yingyan.li/repo/WoTE/dataset/maps"
export NAVSIM_EXP_ROOT="/home/yingyan.li/repo/WoTE/exp"
export NAVSIM_DEVKIT_ROOT="/home/yingyan.li/repo/WoTE/"
export OPENSCENE_DATA_ROOT="/home/yingyan.li/repo/WoTE/dataset"

# After (replace with YOUR paths)
export PYTHONPATH=/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/
export NUPLAN_MAPS_ROOT="/home/yingyan.li/data/navsim/maps"  # Use shared dataset
export NAVSIM_EXP_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/exp"
export NAVSIM_DEVKIT_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/"
export OPENSCENE_DATA_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project"
```

### 4. `scripts/miscs/k_means_trajs.py`
**Changes:** Fixed hardcoded paths and sklearn API issues

**Key fixes:**
- Use environment variables instead of hardcoded paths
- Changed `kmeans.trajectory_anchors_` to `kmeans.cluster_centers_` (correct sklearn attribute)
- Added directory creation for output
- Added file existence check to skip re-collecting data

```python
# Before (lines with hardcoded paths)
data_root = '/home/yingyan.li/repo/WoTE/dataset/'
repo_root = '/home/yingyan.li/repo/WoTE/'

# After
data_root = os.environ.get('OPENSCENE_DATA_ROOT', '/scratch/.../rob535-final-project')
repo_root = os.environ.get('NAVSIM_DEVKIT_ROOT', '/scratch/.../rob535-final-project/')

# Before (incorrect sklearn API)
trajectory_anchors_256 = kmeans.trajectory_anchors_

# After
trajectory_anchors_256 = kmeans.cluster_centers_
```

### 5. `scripts/miscs/gen_pdm_score.sh`
**Changes:** Updated paths to use your scratch directory

```bash
# Before
'agent.checkpoint_path="/home/yingyan.li/repo/WoTE/exp/..."'
metric_cache_path='/home/yingyan.li/repo/WoTE/exp/metric_cache/trainval'

# After
'agent.checkpoint_path="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/exp/..."'
metric_cache_path='/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/exp/metric_cache/trainval'
```

### 6. `scripts/miscs/gen_multi_trajs_pdm_score.py`
**Changes:** Fixed hardcoded paths in three locations

**Lines modified:**
- Line 47-49: Hydra config path (use relative path from script)
- Line 147: Save path for formatted PDM scores
- Line 242: Load path for trajectory anchors

```python
# Before (line 47)
CONFIG_PATH = "/home/yingyan.li/repo/WoTE/navsim/planning/script/config/pdm_scoring"

# After
REPO_ROOT = os.path.join(os.path.dirname(__file__), '../..')
CONFIG_PATH = os.path.join(REPO_ROOT, 'navsim/planning/script/config/pdm_scoring')

# Before (line 147)
save_path = f'/home/yingyan.li/repo/WoTE/dataset/extra_data/planning_vb/formatted_pdm_score_{num_clusters}.npy'

# After
data_root = os.environ.get('OPENSCENE_DATA_ROOT', os.path.join(os.path.dirname(__file__), '../..'))
save_path = os.path.join(data_root, f'extra_data/planning_vb/formatted_pdm_score_{num_clusters}.npy')
```

---

## Files Created

### 1. `HPC_SETUP.md`
**Purpose:** Comprehensive guide for setting up WoTE on Great Lakes HPC

**Key sections:**
- Initial setup and conda environment creation
- Setup script for environment variables
- Data organization and symlink structure
- VS Code integration via Open OnDemand
- SLURM job submission
- Cost estimation and troubleshooting

### 2. `TRAINING_GUIDE.md`
**Purpose:** Step-by-step training instructions with cost estimates

**Key sections:**
- Pre-training checklist
- Quick test run (1 epoch, 10 minutes)
- Full training options (interactive vs batch)
- Monitoring and troubleshooting
- Cost breakdowns

### 3. `setup_wote.sh` (in `/home/<uniqname>/`)
**Purpose:** Environment setup script for all sessions

```bash
#!/bin/bash

# Load required modules
module load python3.9-anaconda

# Initialize conda (enables 'conda activate' command)
eval "$(conda shell.bash hook)"

# Activate your conda environment
conda activate wote

# Fix PATH to ensure wote Python is used (not base environment)
export PATH=/home/<uniqname>/.conda/envs/wote/bin:$PATH

# Set environment variables for NAVSIM
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/yingyan.li/data/navsim/maps"
export NAVSIM_EXP_ROOT="/scratch/.../rob535-final-project/exp"
export NAVSIM_DEVKIT_ROOT="/scratch/.../rob535-final-project/"
export OPENSCENE_DATA_ROOT="/scratch/.../rob535-final-project"
export PYTHONNOUSERSITE=1  # Ignore ~/.local packages that may conflict

# Proxy for extensions (if needed)
source /etc/profile.d/http_proxy.sh
```

### 4. `slurm_jobs/train_1epoch_test.slurm`
**Purpose:** SLURM script for 1-epoch validation run

**Key features:**
- 2-hour time limit
- Single GPU
- Saves to separate experiment folder
- Creates logs with job ID

### 5. `.gitignore` updates
**Added entries:**
```
# HPC specific
logs/
slurm-*.out
*.slurm.out
*.slurm.err

# Data and models
ckpts/
extra_data/
exp/
dataset/
maps/
*_navsim_logs/
*_sensor_blobs/
navsim_logs/
sensor_blobs/

# Environment
.conda/
```

---

## Step-by-Step Setup Process

### Phase 1: Environment Setup (1-2 hours)

1. **Clone repo on Great Lakes:**
   ```bash
   cd /scratch/<account_root>/<account>/<uniqname>
   git clone <repo_url> rob535-final-project
   cd rob535-final-project
   ```

2. **Modify `environment.yml`:**
   - Change `python=3.8` to `python=3.9`

3. **Create conda environment (use compute node!):**
   ```bash
   salloc --account=<account> --partition=standard --nodes=1 --cpus-per-task=4 --mem=16GB --time=02:00:00
   module load python3.9-anaconda
   conda init bash
   source ~/.bashrc
   cd /scratch/.../rob535-final-project
   conda env create -f environment.yml
   conda activate wote
   pip install -r requirements.txt
   pip install git+https://github.com/motional/nuplan-devkit.git@nuplan-devkit-v1.2#egg=nuplan-devkit
   pip install -e .
   exit
   ```

4. **Create setup script:**
   - Create `/home/<uniqname>/setup_wote.sh` with proper paths
   - `chmod +x ~/setup_wote.sh`

### Phase 2: Data Structure Setup (30 minutes)

1. **Create directories:**
   ```bash
   cd /scratch/.../rob535-final-project
   mkdir -p exp/metric_cache
   mkdir -p ckpts
   mkdir -p extra_data/planning_vb
   mkdir -p logs
   mkdir -p slurm_jobs
   ```

2. **Create symlinks to shared dataset:**
   ```bash
   # Check what's available
   ls /home/yingyan.li/data/navsim/
   
   # Create symlinks based on actual structure
   # Example (adjust based on what you find):
   ln -s /home/yingyan.li/data/navsim/trainval_navsim_logs .
   ln -s /home/yingyan.li/data/navsim/trainval_sensor_blobs .
   ln -s /home/yingyan.li/data/navsim/test_navsim_logs .
   ln -s /home/yingyan.li/data/navsim/test_sensor_blobs .
   
   # Create nested structure for code expectations
   mkdir -p navsim_logs sensor_blobs
   ln -s ../trainval_navsim_logs/trainval navsim_logs/trainval
   ln -s ../trainval_sensor_blobs/trainval sensor_blobs/trainval
   ```

### Phase 3: Update Config Files (30 minutes)

1. **Update `navsim/agents/WoTE/configs/default.py`:**
   - Lines 16, 136, 137 with your scratch paths

2. **Update `scripts/training/run_wote.sh`:**
   - All environment variable exports

3. **Update `scripts/miscs/k_means_trajs.py`:**
   - Replace hardcoded paths with environment variables
   - Fix `trajectory_anchors_` → `cluster_centers_`
   - Add directory creation and file checks

4. **Update `scripts/miscs/gen_multi_trajs_pdm_score.py`:**
   - Lines 47-49, 147, 242

5. **Update `scripts/miscs/gen_pdm_score.sh`:**
   - All paths to your scratch directory

### Phase 4: Generate/Download Required Files (1-2 hours)

1. **Generate trajectory anchors:**
   ```bash
   # Request compute node
   salloc --account=<account> --partition=standard --nodes=1 --cpus-per-task=8 --mem=32GB --time=02:00:00
   
   # Set up environment
   source ~/setup_wote.sh
   cd /scratch/.../rob535-final-project
   
   # Run k-means clustering
   python scripts/miscs/k_means_trajs.py
   
   # Verify output
   ls -lh extra_data/planning_vb/trajectory_anchors_256.npy  # Should be ~25KB
   ```

2. **Download pre-computed files from Google Drive:**
   ```bash
   # Install gdown if needed
   pip install gdown
   
   # Download formatted_pdm_score_256.npy
   cd /scratch/.../rob535-final-project
   gdown 1STElIeiY7rQ4QboWuyro5IirVUZhHSTm -O extra_data/planning_vb/formatted_pdm_score_256.npy
   
   # Download resnet34.pth
   gdown 1xbz52Zev2ThG9JO1VPUuPDgb0iQnRcD5 -O ckpts/resnet34.pth
   
   # Verify downloads
   ls -lh extra_data/planning_vb/formatted_pdm_score_256.npy  # Should be ~1.5GB
   ls -lh ckpts/resnet34.pth  # Should be ~84MB
   ```

### Phase 5: Test Training (1-2 hours)

1. **Quick interactive test:**
   ```bash
   salloc --account=<account> --partition=gpu --nodes=1 --cpus-per-task=8 --mem=32GB --gres=gpu:1 --time=01:00:00
   source ~/setup_wote.sh
   cd /scratch/.../rob535-final-project
   
   # Run short test
   python ./navsim/planning/script/run_training.py \
   agent=WoTE_agent \
   agent.config._target_=navsim.agents.WoTE.configs.default.WoTEConfig \
   experiment_name=WoTE/test_run \
   scene_filter=navtrain \
   dataloader.params.batch_size=4 \
   trainer.params.max_epochs=1 \
   split=trainval
   ```

2. **Submit 1-epoch validation job:**
   ```bash
   # Copy slurm script template (update paths inside)
   sbatch slurm_jobs/train_1epoch_test.slurm
   squeue -u <uniqname>
   tail -f logs/train_1epoch_test_<job_id>.out
   ```

### Phase 6: Full Training (20-40 hours)

Once validation succeeds, submit full training job with 30 epochs.

---

## Key Configuration Changes

### Environment Variables Required

These must be set in every session (via `setup_wote.sh` or manually):

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/home/yingyan.li/data/navsim/maps"
export NAVSIM_EXP_ROOT="/scratch/.../rob535-final-project/exp"
export NAVSIM_DEVKIT_ROOT="/scratch/.../rob535-final-project/"
export OPENSCENE_DATA_ROOT="/scratch/.../rob535-final-project"
export PYTHONNOUSERSITE=1
export PATH=/home/<uniqname>/.conda/envs/wote/bin:$PATH
```

### Python Environment Activation

**Critical workflow** (conda activation doesn't work by default on Great Lakes):

```bash
module load python3.9-anaconda
eval "$(conda shell.bash hook)"  # THIS IS ESSENTIAL
conda activate wote
export PATH=/home/<uniqname>/.conda/envs/wote/bin:$PATH  # THIS TOO
```

### Data Structure Expected by Code

```
/scratch/.../rob535-final-project/
├── navsim_logs/
│   └── trainval/ → ../trainval_navsim_logs/trainval/
├── sensor_blobs/
│   └── trainval/ → ../trainval_sensor_blobs/trainval/
├── trainval_navsim_logs/
│   └── trainval/
│       └── (actual log files)
├── trainval_sensor_blobs/
│   └── trainval/
│       └── (actual sensor files)
```

Note the **nested trainval directories** - this is required!

---

## Downloads Required

### From Google Drive

**Folder:** https://drive.google.com/drive/folders/1dIHK8nXkzhIhGCRQOpKibaizwH-7fHqs

**Required files:**

1. **`formatted_pdm_score_256.npy`** (~1.5GB)
   - File ID: `1STElIeiY7rQ4QboWuyro5IirVUZhHSTm`
   - Location: `extra_data/planning_vb/`
   - Purpose: Pre-computed trajectory reward scores for training

2. **`resnet34.pth`** (~84MB)
   - File ID: `1xbz52Zev2ThG9JO1VPUuPDgb0iQnRcD5`
   - Location: `ckpts/`
   - Purpose: Pre-trained ResNet34 backbone weights

**Download commands:**
```bash
pip install gdown
cd /scratch/.../rob535-final-project

# Download both files
gdown 1STElIeiY7rQ4QboWuyro5IirVUZhHSTm -O extra_data/planning_vb/formatted_pdm_score_256.npy
gdown 1xbz52Zev2ThG9JO1VPUuPDgb0iQnRcD5 -O ckpts/resnet34.pth
```

### Generated Locally

**`trajectory_anchors_256.npy`** (~25KB)
- Generated by: `python scripts/miscs/k_means_trajs.py`
- Location: `extra_data/planning_vb/`
- Purpose: 256 trajectory templates from K-means clustering
- Time to generate: ~10-15 minutes on 8-core node
- Cost: ~$0.10-0.15

---

## Common Issues Fixed

### Issue 1: Conda activate doesn't work
**Symptom:** `conda: command not found` or `conda activate` does nothing

**Solution:**
```bash
module load python3.9-anaconda
eval "$(conda shell.bash hook)"
conda activate wote
```

### Issue 2: Wrong Python is used (base instead of wote)
**Symptom:** `ModuleNotFoundError` for packages you installed

**Solution:**
```bash
export PATH=/home/<uniqname>/.conda/envs/wote/bin:$PATH
which python  # Verify it shows /home/<uniqname>/.conda/envs/wote/bin/python
```

### Issue 3: Package conflicts with ~/.local
**Symptom:** Version conflicts, import errors

**Solution:**
```bash
export PYTHONNOUSERSITE=1
```

### Issue 4: Data structure mismatch
**Symptom:** `FileNotFoundError` for data files

**Solution:**
- Check actual structure of shared dataset: `ls -la /home/yingyan.li/data/navsim/`
- Create nested symlinks: `navsim_logs/trainval/` must point to actual data location
- Pattern is usually: `trainval_navsim_logs/trainval/` (nested trainval directories)

### Issue 5: tensorboard incompatible with Python 3.8
**Symptom:** Installation fails with version conflict

**Solution:**
- Change `environment.yml` to `python=3.9`
- Recreate conda environment

### Issue 6: sklearn API changed
**Symptom:** `AttributeError: 'MiniBatchKMeans' has no attribute 'trajectory_anchors_'`

**Solution:**
- Change to `kmeans.cluster_centers_` (correct sklearn attribute)

### Issue 7: Out of space during pip install
**Symptom:** `No space left on device` during pip install

**Solution:**
- Usually temporary filesystem congestion
- Wait and retry, or use compute node with more resources

### Issue 8: Training killed at 1 hour
**Symptom:** Job killed before epoch completes

**Solution:**
- 1 epoch takes ~1.5 hours on single GPU
- Request at least 2 hours: `--time=02:00:00`
- For 30 epochs: request 48-50 hours

---

## Cost Estimates (Great Lakes GPU Partition)

- **Test run (1 epoch, 2 hours):** ~$0.33
- **Full training (30 epochs, 48 hours):** ~$8-10
- **8 GPU training (3-6 hours):** ~$36-96

**Free allocation:** $60 per project member

---

## Quick Reference Commands

### Start new session:
```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu
cd /scratch/.../rob535-final-project
source ~/setup_wote.sh
```

### Estimate job cost:
```bash
my_job_estimate -p gpu -n 1 -c 8 -m 32g -g 1 -t HH:MM:SS
```

### Submit training job:
```bash
sbatch slurm_jobs/train_1epoch_test.slurm
squeue -u <uniqname>
tail -f logs/train_*_<job_id>.out
```

### Verify setup:
```bash
which python  # Should show wote env
python -c "import torch; print(torch.__version__)"  # Should be 2.0.1+cu118
ls -lh extra_data/planning_vb/trajectory_anchors_256.npy
ls -lh extra_data/planning_vb/formatted_pdm_score_256.npy
ls -lh ckpts/resnet34.pth
```

---

## Summary Checklist

Before starting training, verify:

- [ ] Conda environment `wote` created with Python 3.9
- [ ] All config files updated with your scratch paths
- [ ] Data symlinks created (check nested trainval directories)
- [ ] `trajectory_anchors_256.npy` exists (25KB)
- [ ] `formatted_pdm_score_256.npy` downloaded (1.5GB)
- [ ] `resnet34.pth` downloaded (84MB)
- [ ] Setup script created: `/home/<uniqname>/setup_wote.sh`
- [ ] Test training runs successfully (interactive or 1-epoch job)
- [ ] Checkpoint saves correctly

---

## Additional Resources

- **HPC_SETUP.md:** Detailed HPC setup instructions
- **TRAINING_GUIDE.md:** Training workflow and monitoring
- **Great Lakes Documentation:** https://arc.umich.edu/greatlakes/
- **Original WoTE Repo:** https://github.com/BraveGroup/WoTE

---

## Notes for Your Friends

1. **Replace ALL paths** with your own scratch directory paths
2. **Check the shared dataset location** - it may have moved or changed
3. **Budget wisely** - monitor costs with `my_job_estimate` before every job
4. **Start with test runs** - don't jump straight to 30 epochs
5. **Save your work** - checkpoints are in `exp/WoTE/*/lightning_logs/version_*/checkpoints/`
6. **Ask for help early** - setup issues compound quickly

Good luck! 🚀
