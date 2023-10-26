#!/bin/bash
<PATH-TO-CONDA>/bin/activate vmf

# Set the -e option. Exit on first error
set -e

RAW_DATA_PATH=<PATH-TO-WAYMO-OPEN-DATASET>/uncompressed/scenario
OUTPUT_PATH=<PATH-TO-PAMLB>/VMF/data/waymo
python data_preprocess.py $RAW_DATA_PATH  $OUTPUT_PATH
