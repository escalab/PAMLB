# VQImg (Orig: Visual Queries 2D localization)

[Original source code](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D)

## Installation instructions
1. Clone the source code.

	```
	git clone https://github.com/escalab/PAMLB.git
	cd VQImg/
	export VQIMG_ROOT=$PWD
	```
2. Create `conda` environment.

	```
	conda create -n vqimg python=3.8
	conda activate vqimg
	```
3. Install [PyTorch with GPU support](https://pytorch.org/). Our experiments rely on CUDA 12.1, the latest version provided by the official PyTorch website was CUDA 11.8 as of 06/23/2023. Therefore, we built the PyTorch nightly version.

	```
	pip install torch==2.1.0.dev20230501+cu121 -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
	pip install torchvision==0.16.0.dev20230501+cu121 -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
	pip install torchaudio==2.1.0.dev20230501+cu121 -f https://download.pytorch.org/whl/nightly/cu121/torch_nightly.html
	```

	For Jetson platform, please check [VQImg-Jetson-Orin.md](./VQImg-Jetson-Orin.md).

4. Install additional requirements using `pip`.

	```
	pip install -r requirements.txt
	```
5. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)

	```
	python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
	```
6. Install pytracking according to [these instructions](https://github.com/visionml/pytracking/blob/master/INSTALL.md). Download the pre-trained [KYS tracker weights](https://drive.google.com/drive/folders/1WGNcats9lpQpGjAmq0s0UwO6n22fxvKi) to `$VQIMG_ROOT/pretrained_models/kys.pth`.
	
	```
	mkdir -p $VQIMG_ROOT/pretrained_models
	# put kys.pth into pretrained_models
	```

	6.1 Install necessary dependencies for PyTracking. [Original source](https://github.com/visionml/pytracking/blob/master/INSTALL.md)
	
	```
	mkdir -p dependencies
	cd $VQIMG_ROOT/dependencies
	git clone https://github.com/visionml/pytracking.git
	cd pytracking/
	git checkout de9cb9bb4f8cad98604fe4b51383a1e66f1c45c0
	```
	6.2 Install dependencies.
	
	```
	conda install matplotlib pandas tqdm
	pip install opencv-python visdom tb-nightly scikit-image tikzplotlib gdown
	pip install Ninja
	```
	6.3 Install the coco and lvis toolkits.
	
	```
	conda install cython
	pip install pycocotools lvis
	```
	6.4 Install ninja-build for Precise ROI pooling.
	
	To compile the [Precise ROI pooling module](https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.
	```
	sudo apt-get install ninja-build
	```
	
	In case of issues, we refer to [https://github.com/vacancy/PreciseRoIPooling](https://github.com/vacancy/PreciseRoIPooling).
	
	6.5 Install spatial-correlation-sampler (only required for KYS tracker), for more information [spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension).
	
	```
	pip install spatial-correlation-sampler
	```
	6.6 Install jpeg4py.
	
	```
	sudo apt-get install libturbojpeg
	pip install jpeg4py 
	```
	6.7 Setup `pytracking` environment.
	
	```
	# Note: In $VQIMG_ROOT/dependencies/pytracking directory.
	# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
	python3 -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
	
	# Environment settings for ltr. Saved at ltr/admin/local.py
	python3 -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
	```
7. Download [Config](https://dl.fbaipublicfiles.com/ego4d/model_zoo/vq2d/slurm_8gpus_4nodes_baseline/config.yaml) and [Checkpoint](https://dl.fbaipublicfiles.com/ego4d/model_zoo/vq2d/slurm_8gpus_4nodes_baseline/model.pth), trained with VQ2D v1.0. Original source and the latest checkpoint can be found at [vq2d_cvpr](https://github.com/facebookresearch/vq2d_cvpr).

	```
	mkdir -p experiments/experiment1/logs
	# put downloaded model.pth and config.yaml to $VQIMG_ROOT/experiments/experiment1/logs/
	```

8. We provide the `enable_em_vqimg-x86.sh` script to set necessary environment variables.

	```
	#!/usr/bin/bash
	
	# Add conda path
	export PATH="$PATH:<PATH-TO-CONDA>/bin"
	source activate vqimg
	
	CUDA_DIR=/usr/local/cuda
	
	# Add cuda, cudnn paths
	export CUDA_HOME=$CUDA_DIR
	export CUDNN_INCLUDE_DIR="$CUDA_HOME/include"
	export CUDNN_LIBRARY="$CUDA_HOME/lib64"
	export CUDACXX=$CUDA_HOME/bin/nvcc
	
	# Add directory paths for VQImg
	export VQIMG_ROOT=<PATH-TO-PAMLB>/VQImg
	export CLIPS_ROOT=$VQIMG_ROOT/data/test_clips
	export VQ2D_SPLITS_ROOT=$VQIMG_ROOT/data
	export EXPT_ROOT=$VQIMG_ROOT/experiments/experiment1
	export PYTRACKING_ROOT=$VQIMG_ROOT/dependencies/pytracking
	```

## Data preparation for training and inference
Note 1: We use the version v1.0 of the VQ2D videos/annotations because we only focus on the inference part. To avoid the insufficient storage capacity on the Jetson Orin, we will skip the full dataset download/preprocess parts (Step 1-4) and provide with two processed clips and simplified annotation cropped from the original `vq_test_unannotated.json` for the demo purpose (Step 5). 

Note 2: We skip training related steps in this demo. Please find the original step-by-step instructions [here](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D#preparing-data-for-training-and-inference) for the complete process.

1. Download the videos as instructed [here](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) to `$VQIMG_ROOT/data`.

	```
	ego4d --output_directory="$VQIMG_ROOT/data" --datasets full_scale
	# Define ego4d videos directory
	export EGO4D_VIDEOS_DIR=$VQIMG_ROOT/data/v1/full_scale
	```

2. Download the annotations to `VQIMG_ROOT/data`.

	```
	# Download the data using the Ego4D CLI.
	ego4d --output_directory="$VQIMG_ROOT/data" --datasets annotations -y --version v1
	
	# Move out vq annotations to $VQIMG_ROOT/data
	mv $VQIMG_ROOT/data/v1/annotations/vq_*.json $VQIMG_ROOT/data
	```
3. Process the VQ dataset.

	```
	python3 process_vq_dataset.py --annot-root data --save-root data
	```
4. Extract clips for val and test data from videos.

	```
	# Extract clips (should take 12-24 hours on a machine with 80 CPU cores)
	python3 convert_videos_to_clips.py \
    --annot-paths data/vq_val.json data/vq_test_unannotated.json \
    --save-root data/clips \
    --ego4d-videos-root $EGO4D_VIDEOS_DIR \
    --num-workers 10 # Increase this for speed
    
    # Validate the extracted clips (should take 30 minutes)
	python3 tools/validate_extracted_clips.py \
    --annot-paths data/vq_val.json data/vq_test_unannotated.json \
    --clips-root data/clips
	```

5. Evaluate models

	(1) Extracting per-frame bbox proposals.
	
	```
	# Note: MODEL_ROOT and DETECTIONS_SAVE_ROOT must be absolute paths, edit these paths in run_extraction-x86.sh and run_infer_vq-x86.sh
	# MODEL_ROOT=<PATH-TO-PRETRAINED-MODEL>  # contains model.pth and config.yaml
	# DETECTIONS_SAVE_ROOT=<PATH-TO-SAVED-PRE-COMPUTED-DETECTIONS>
	
	cd $VQIMG_ROOT
	
	# Extract per-frame bbox proposals and visual query similarity scores
	# Edit PATH-TO-VQIMG in run_extraction-x86.sh
	chmod +x ./run_extraction-x86.sh
	./run_extraction-x86.sh
	```
	
	(2) Peak detection and bidirectional tracking.
	
	```
	# Edit PATH-TO-VQIMG in run_infer_vq-x86.sh
	chmod +x ./run_infer_vq-x86.sh
	./run_infer_vq-x86.sh
	```

	(3) Parse the results

	```
	./run_infer_vq-x86.sh > vqimg_output.txt
	python vqimg_parser.py --file vqimg_output.txt
	```

## Hardware/software specifications
- NVIDIA RTX 3090 GPU
	- CUDA 12.1
	- PyTorch v2.1.0.dev20230501+cu121
	- torchvision v0.16.0.dev20230501+cu121
- Ubuntu 20.04
- Detectron2 v0.6

## Troubleshooting
Please check [Troubleshooting Guide](./TROUBLESHOOTING.md) for the potential errors you may encounter during the reproduction.

## Acknowledgements
Original codebase from [episodic-memory/VQ2D](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D#preparing-data-for-training-and-inference), relying on [detectron2](https://github.com/facebookresearch/detectron2), [vq2d_cvpr](https://github.com/facebookresearch/vq2d_cvpr), [PyTracking](https://github.com/visionml/pytracking), [pfilter](https://github.com/johnhw/pfilter), and [ActivityNet](https://github.com/activitynet/ActivityNet) repositories. We utilized the VQ2D codebase to measure the performance of emerging personal assistant applications and generated insights for different data processing models.