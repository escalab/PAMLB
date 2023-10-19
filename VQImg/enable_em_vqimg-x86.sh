#!/usr/bin/bash

# Add conda path
export PATH="$PATH:<PATH-TO-CONDA>/bin"
# Activate conda environment
source activate vqimg

CUDA_DIR=/usr/local/cuda
# CUDNN_DIR=/usr/lib/aarch64-linux-gnu

# Add cuda, cudnn paths
export CUDA_HOME=$CUDA_DIR
# export CUDNN_PATH=$CUDNN_DIR
export CUDNN_INCLUDE_DIR="$CUDA_HOME/include"
export CUDNN_LIBRARY="$CUDA_HOME/lib64"
export CUDACXX=$CUDA_HOME/bin/nvcc

# Add directory paths for VQImg
export VQIMG_ROOT="<PATH-TO-PAMLB>/VQImg"
export CLIPS_ROOT=$VQIMG_ROOT/data/test_clips
export VQ2D_SPLITS_ROOT=$VQIMG_ROOT/data
export EXPT_ROOT=$VQIMG_ROOT/experiments/experiment1
export PYTRACKING_ROOT=$VQIMG_ROOT/dependencies/pytracking
