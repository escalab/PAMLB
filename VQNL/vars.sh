#!/bin/bash
export NAME=official_v1
export TASK_NAME=nlq_$NAME
export BASE_DIR=data/dataset/nlq_$NAME
export FEATURE_BASE_DIR=data/features/nlq_$NAME
export FEATURE_DIR=$FEATURE_BASE_DIR/video_features
export VQNL_ROOT=<PATH-TO-PAMLB>/VQNL
export MODEL_BASE_DIR=$VQNL_ROOT/checkpoints

