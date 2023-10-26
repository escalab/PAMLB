#!/bin/bash
<PATH-TO-CONDA>/bin/activate vmf

# Set the -e option. Exit on first error
set -e

python train.py --cfg_file cfgs/waymo/mtr+20_percent_data.yaml \
       --extra_tag my_first_exp \
       --batch_size 8 \
       --epochs 10
