#!/bin/bash
source vars.sh

python utils/prepare_ego4d_dataset.py \
    --input_train_split data/nlq_train.json \
    --input_val_split data/nlq_val.json \
    --input_test_split data/nlq_test_unannotated.json \
    --video_feature_read_path $FEATURE_DIR \
    --clip_feature_save_path $FEATURE_BASE_DIR/official \
    --output_save_path $BASE_DIR
