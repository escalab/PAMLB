#!/bin/bash
<PATH-TO-CONDA>/bin/activate vqnl
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
source vars.sh

# machine parameters
export DATALOADER_WORKERS=1
export NUM_WORKERS=2
export VAL_JSON_PATH="data/nlq_val.json"

# hyper parameters
export BATCH_SIZE=32
export DIM=128
export VIDEO_FEATURE_DIM=1536
export NUM_EPOCH=20
export MAX_POS_LEN=128
export INIT_LR=0.0025

export TB_LOG_NAME="${NAME}_bs${BATCH_SIZE}_dim${DIM}_epoch${NUM_EPOCH}_ilr${INIT_LR}"

# Use a single GPU to train the model by default.
python main.py \
    --gpu_idx 0 \
    --task $TASK_NAME \
    --predictor bert \
    --dim $DIM \
    --mode train \
    --video_feature_dim $VIDEO_FEATURE_DIM \
    --max_pos_len $MAX_POS_LEN \
    --init_lr $INIT_LR \
    --epochs $NUM_EPOCH \
    --batch_size $BATCH_SIZE \
    --fv official \
    --num_workers $NUM_WORKERS \
    --data_loader_workers $DATALOADER_WORKERS \
    --model_dir $MODEL_BASE_DIR/$NAME \
    --eval_gt_json $VAL_JSON_PATH \
    --log_to_tensorboard "baseline" \
    --tb_log_freq 5 \
    --remove_empty_queries_from train
