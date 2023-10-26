#!/bin/bash
<PATH-TO-CONDA>/bin/activate vmf

# Set the -e option. Exit on first error
set -e

bash scripts/dist_train.sh 8 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml \
       --extra_tag my_first_exp \
       --batch_size 64 \
       --epochs 30
