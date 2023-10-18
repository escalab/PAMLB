#!/bin/bash

source /root/workplace/episodic-memory/VQ2D/enable_em_vq2d.sh

export MODEL_ROOT="$VQIMG_ROOT/experiments/experiment1/logs"
export DETECTIONS_SAVE_ROOT="$VQIMG_ROOT"
export SPLIT="test_one"
export CACHE_ROOT="$VQIMG_ROOT"

cd $VQIMG_ROOT
export PYTHONPATH="/usr/bin/python3:$VQIMG_ROOT:$PYTRACKING_ROOT"

python3 -W ignore extract_vq_detection_scores.py \
  data.data_root="$CLIPS_ROOT" \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.split="$SPLIT" \
  data.num_processes_per_gpu=1 \
  data.rcnn_batch_size=10 \
  model.config_path="$MODEL_ROOT/config.yaml" \
  model.checkpoint_path="$MODEL_ROOT/model.pth" \
  model.cache_root="$CACHE_ROOT"
