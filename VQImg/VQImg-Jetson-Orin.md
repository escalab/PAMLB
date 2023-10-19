# VQImg for Jetson Orin

[VQ2D original source code and installation steps (x86)](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D)

To install PyTorch locally on Jetson platform, please follow instructions on [NVIDIA DOCS](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html). We suggest the use of [NVIDIA L4T PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch) container to avoid library dependencies issue. We use `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` on our Jetson Orin with JetPack 5.1.1.

## Installation instructions (with L4T PyTorch container)

1. Pull `l4t-pytorch` container tags (corresponding to the version of JetPack-L4T on your Jetson).

	```
	sudo docker pull nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
	```

2. Clone the source code. (Use NFS `Docker Volumes` as an example)

	```
	git clone https://github.com/escalab/PAMLB.git
	cd VQImg/
	```

	```
	# create Docker volume
	docker volume create --driver local \ 
	--opt type=nfs \ 
	--opt o=addr=[ip-address] \ 
	--opt device=:[path-to-directory] [volume-name]
	
	# check the volume
	docker volume inspect[volume-name]
	```

	For example, 

	```
	docker volume create --driver local \
	--opt type=nfs \
	--opt o=addr=[ip-address] \
	--opt device=:/home/PAMLB/VQImg myvqimg
	```
3. Start an interactive session in the container with a volume. The rest of steps will be executed in the container.

	```
	docker run -it --runtime nvidia \
	--network=host \
	--name myvqimg \
	--mount source=myvqimg,target=/mnt/VQImg \
	nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
	```
	
	Note: `--runtime nvidia` to have Torch GPU support

4. Install additional requirements using `pip`.

	```
	cd /mnt/VQImg
	export VQIMG_ROOT=$PWD
	pip install -r requirements.txt
	```
5. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

	```
	mkdir -p dependencies
	cd dependencies/
	git clone https://github.com/facebookresearch/detectron2.git
	cd detectron2/
	python3 setup.py install develop
	```

6. Install pytracking according to [these instructions](https://github.com/visionml/pytracking/blob/master/INSTALL.md). Download the pre-trained [KYS tracker weights](https://drive.google.com/drive/folders/1WGNcats9lpQpGjAmq0s0UwO6n22fxvKi) to `$VQIMG_ROOT/pretrained_models/kys.pth`.

	```
	mkdir -p $VQIMG_ROOT/pretrained_models
	# put kys.pth into pretrained_models
	```
	
	6.1 Install necessary dependencies for PyTracking [Original source](https://github.com/visionml/pytracking/blob/master/INSTALL.md).
	
	```
	cd $VQIMG_ROOT/dependencies
	git clone https://github.com/visionml/pytracking.git
	cd pytracking/
	git checkout de9cb9bb4f8cad98604fe4b51383a1e66f1c45c0
	```
	6.2 Install dependencies.
	
	```
	pip install matplotlib tqdm opencv-python
	# pip install tb-nightly tikzplotlib gdown
	```
	6.3 Install the coco and lvis toolkits.
	
	```
	pip install cython
	pip install pycocotools lvis
	```
	6.4 Install ninja-build for Precise ROI pooling.
	
	To compile the [Precise ROI pooling module](https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.
	```
	sudo apt-get install ninja-build
	```
	
	In case of issues, we refer to [https://github.com/vacancy/PreciseRoIPooling](https://github.com/vacancy/PreciseRoIPooling).
	
	6.5 Install spatial-correlation-sampler (only required for KYS tracker) dependency for pytracking. More information [spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension).
	
	```
	pip install spatial-correlation-sampler
	```
	6.6 Install jpeg4py.
	
	```
	sudo apt-get install libturbojpeg
	pip install jpeg4py 
	```
	
	Note: If the pip install for [spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension) fails:
	
	```
	cd $VQIMG_ROOT/dependencies
	git clone git@github.com:ClementPinard/Pytorch-Correlation-extension.git
	cd Pytorch-Correlation-extension
	python3 setup.py install
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
	
8. We provide the `enable_em_vqimg-arm64.sh` script to set necessary environment variables.

	```
	#!/usr/bin/bash
	
	# Add CUDA, CuDNN paths
	CUDA_DIR=<PATH-TO-CUDA>
	CUDNN_DIR=<PATH-TO-CUDNN>
	
	export CUDA_HOME=$CUDA_DIR
	export CUDNN_INCLUDE_DIR="/usr/include"
	export CUDNN_LIBRARY=$CUDNN_DIR
	export CUDACXX=$CUDA_DIR/bin/nvcc
	
	# Add directory paths for VQImg
	export VQIMG_ROOT="<PATH-TO-VQIMG>"
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
	# Note: MODEL_ROOT and DETECTIONS_SAVE_ROOT must be absolute paths, edit these paths in run_extraction-arm64.sh and run_infer_vq-arm64.sh
	# MODEL_ROOT=<PATH-TO-PRETRAINED-MODEL>  # contains model.pth and config.yaml
	# DETECTIONS_SAVE_ROOT=<PATH-TO-SAVED-PRE-COMPUTED-DETECTIONS>
	
	cd $VQIMG_ROOT
	
	# Extract per-frame bbox proposals and visual query similarity scores
	# Edit PATH-TO-VQIMG in run_extraction-arm64.sh
	chmod +x ./run_extraction-arm64.sh
	./run_extraction-arm64.sh
	```
	
	(2) Peak detection and bidirectional tracking.
	
	```
	# Edit PATH-TO-VQIMG in run_infer_vq-arm64.sh
	chmod +x ./run_infer_vq-arm64.sh
	./run_infer_vq-arm64.sh
	```

## Hardware/software specifications
- NVIDIA Orin NX Developer Kit (8GB)
- Jetpack 5.1.1 [L4T 35.3.1]
- l4t-pytorch:r35.2.1-pth2.0-py3
	- PyTorch v2.0.0
	- torchvision v0.14.1
	- torchaudio v0.13.1
- Detectron2 v0.6+cu111
