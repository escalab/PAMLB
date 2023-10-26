# Text-to-image generation

[Original source code](https://github.com/sshaoshuai/MTR/tree/6d89ad8b59f2b8791b6b924bf272797b0529467c). Motion Transformer (MTR): A Strong Baseline for Multimodal Motion Prediction in Autonomous Driving

## Installation instructions
1. Clone the source code.

	```
	git clone https://github.com/escalab/PAMLB.git
	cd VMF/
	```

2. Create `conda` environment.

	```
	conda create -n vmf python=3.8
	conda activate vmf
	```

3. Environment setup

	```
	pip install -r requirements.txt
	
	# Compile the customized CUDA codes
	python setup.py develop
	```	
	
	Note: Make sure your torch has GPU support.
	
## Data preparation

1. Download [Waymo Open Dataset](https://waymo.com/open/download/), we use the `scenario protocal` from `waymo_open_dataset_motion_v_1_2_0`(v1.2), the data organization as follows:

	```md
	waymo_motion/uncompressed
				├── occupancy_flow_challenge
				├── tf_example
				└── scenario (~620 GB)
				       └── scenario (~620 GB)
				               ├── testing
				               ├── testing_interactive
				               ├── training
				               ├── training_20s
				               ├── validation
				               └── validation_interactive
	```

	Note: As raw data size exceeds the storage capacity on our Jetson platform, we performed the data preprocessing/training on the server and copy the processed data/pretrained model to the Jetson platform.

	Original data preprocessing can be found [here](https://github.com/sshaoshuai/MTR/blob/6d89ad8b59f2b8791b6b924bf272797b0529467c/docs/DATASET_PREPARATION.md). More information can be found on [waymo\_open\_dataset](https://www.tensorflow.org/datasets/catalog/waymo_open_dataset).
	
2. Install the Waymo Open Dataset API

	```
	pip install waymo-open-dataset-tf-2-6-0
	pip install --upgrade typing-extensions
	``` 

3. Preprocess the dataset

	```
	cd mtr/datasets/waymo
	# set RAW_DATA_PATH (where you store Waymo Open Dataset) and OUTPUT_PATH in `run_data_preprocess.sh` 
	chmod +x ./run_data_preprocess.sh
	./run_data_preprocess.sh
	```
	
	The processed data will be saved to `$OUTPUT_PATH`, e.g., `<PATH-TO-PAMLB>/VMF/data/waymo` as `$OUTPUT_APTH` then the data organization as follows:
	
	
	```md
	VMF
	├── data
	│    └── waymo
	│          ├── processed_scenarios_training
	│          ├── processed_scenarios_validation
	│          ├── processed_scenarios_training_infos.pkl
	│          └── processed_scenarios_val_infos.pkl
	├── mtr
	└── tools
	```
	
	Note: If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (but this will use pure-Python parsing and will be much slower).
	
## Training and Testing
	
1. Use `run_train.sh` to train the model with 1 GPU.

	```
	cd tools
	chmod +x ./run_train.sh
	./run_train.sh
	```
	
	1.1 For multi-GPU training, using `run_train_multi-gpus.sh` script:
	
	```
	chmod +x ./run_train_multi-gpus.sh
	./run_train_multi-gpus.sh
	``` 
	
	The evaluation results will be logged to the log file under `<PATH-TO-PAMLB>/VMF/output/waymo/mtr+20_percent_data/<extra_tag_name>/`.
	
	**Troubleshooting**
	
	```
	# Example error
	torch.cuda.OutOfMemoryError: CUDA out of memory. 
	Tried to allocate 38.00 MiB. GPU 0 has a total capacty of 23.69 GiB 
	of which 34.19 MiB is free...
	```
	Aternatives:
	
	```
	import torch
	# clearing the occupied cuda memory
	torch.cuda.empty_cache()
	```
	
	```
	import gc
	# clear variables that not in use
	del variables
	gc.collect()
	```
	- If the above alternatives do not work, try:
		- Reduce train,val, test data
		- Reduce batch_size, e.g., 8 for a single RTX 3090 GPU
		- Reduce number of model parameters
	
2. Perform inference with 1 GPU using `run_inference.sh` script. 

	
	```
	cd tools
	# set ckpt path '../output/waymo/mtr+20_percent_data/<extra_tag_name>/ckpt/<XXX.pth>'
	# e.g., latest_model.pth
	chmod +x ./run_inference.sh
	./run_inference.sh
	```

3. Parse the performance results.

	```
	./run_inference.sh >& vmf_bs8.txt
	python parser.py --file vmf_bs8.txt
	```

### Other troubleshooting

```
Could not load dynamic library 'libcudart.so.11.0'; 
dlerror: libcudart.so.11.0: cannot open shared object file: 
No such file or directory
```

You need to update the `LD_LIBRARY_PATH` environment variable to include the correct paths to libraries or make sure your tensorflow version has GPU support.

## Hardware/software specifications
- X86
	- NVIDIA RTX 3090 GPU
		- CUDA 12.1
		- PyTorch 2.1.0.dev20230512+cu121
		- MotionTransformer 0.1.0+ac476f6
		- waymo-open-dataset-tf-2-6-0 1.4.9
	- Ubuntu 20.04

- Jetson Orin
	- PyTorch 2.0.0a0+ec3941ad.nv23.2
	- MotionTransformer 0.1.0

## Acknowledgements

```
@article{shi2022motion,
  title={Motion transformer with global intention localization and local movement refinement},
  author={Shi, Shaoshuai and Jiang, Li and Dai, Dengxin and Schiele, Bernt},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```

We utilized MTR as the reference implementation for vehicle motion prediction, measuring the performance, generating insights for different data processing models.
