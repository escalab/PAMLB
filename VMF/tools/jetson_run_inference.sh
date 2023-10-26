#!/bin/bash
#<PATH-TO-CONDA>/bin/activate vmf

# Set the -e option. Exit on first error
set -e

python3 test.py --cfg_file cfgs/waymo/mtr+20_percent_data.yaml \
       --ckpt ../output/waymo/mtr+20_percent_data/my_first_exp/ckpt/latest_model.pth \
       --extra_tag my_first_exp \
       --batch_size 1 \
       --eval_tag eval_exp \
       --save_to_file
