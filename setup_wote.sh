#!/bin/bash

# Load required modules
module load python3.9-anaconda  # or mamba/py3.9 if available
module load cuda/11.8           # For PyTorch 2.0.1 GPU support
module load cudnn/11.8-v8.7.0   # cuDNN for PyTorch

# Activate your conda environment
source activate wote

export PYTHONNOUSERBASE=1

# Set environment variables for NAVSIM
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/maps"
export NAVSIM_EXP_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project/exp"
export NAVSIM_DEVKIT_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project"
export OPENSCENE_DATA_ROOT="/scratch/rob535f25s001_class_root/rob535f25s001_class/abcarr/rob535-final-project"

# Proxy for extensions (if needed)
source /etc/profile.d/http_proxy.sh
