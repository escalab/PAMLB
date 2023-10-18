#!/bin/bash

export VQIMG_ROOT="PATH-TO-VQIMG"
export EGO4D_VIDEOS_DIR="PATH-TO-EGO4D-VIDEOS"

start_time="$(date -u +%s.%N)"
time python3 convert_videos_to_clips.py \
    --annot-paths data/vq_test_one.json \
    --save-root data/clips \
    --ego4d-videos-root $EGO4D_VIDEOS_DIR \
    --num-workers 1 # Increase this for speed, e.g. 10
end_time="$(date -u +%s.%N)"

extract_elapsed="$(bc <<<"$end_time-$start_time")"

echo "Total of $extract_elapsed seconds for clips extraction"
