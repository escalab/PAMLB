#!/bin/bash

source <PATH-TO-VQIMG>/enable_em_vqimg-x86.sh

export MODEL_ROOT="$VQIMG_ROOT/experiments/experiment1/logs"
export DETECTIONS_SAVE_ROOT="$VQIMG_ROOT"
export CACHE_ROOT="$VQIMG_ROOT"
export SPLIT="test_one"
export N_PROCS_PER_GPU="1"
export PEAK_SIMILARITY_THRESH="0.50"
export LOST_THRESH="0.20"

export STATS_PATH="$MODEL_ROOT/${SPLIT}_predictions_pst_${PEAK_SIMILARITY_THRESH}_lost_thresh_${LOST_THRESH}.json"

cd $VQIMG_ROOT

export PYTHONPATH="/usr/bin/python3:$VQIMG_ROOT:$PYTRACKING_ROOT"

NT=1
OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT VECLIB_MAXIMUM_THREADS=$NT NUMEXPR_NUM_THREADS=$NT python3 -W ignore perform_vq_inference.py \
    data.data_root=$CLIPS_ROOT \
    data.annot_root=$VQ2D_SPLITS_ROOT \
    data.split=$SPLIT \
    data.num_processes_per_gpu=$N_PROCS_PER_GPU \
    data.rcnn_batch_size=1 \
    model.config_path=$MODEL_ROOT/config.yaml \
    model.checkpoint_path=$MODEL_ROOT/model.pth \
    model.cache_root=$CACHE_ROOT \
    signals.peak_similarity_thresh=$PEAK_SIMILARITY_THRESH \
    logging.save_dir=$MODEL_ROOT \
    logging.stats_save_path=$STATS_PATH \
    tracker.kys_tracker.model_path=$VQIMG_ROOT/pretrained_models/kys.pth \
    tracker.kys_tracker.lost_thresh=$LOST_THRESH
