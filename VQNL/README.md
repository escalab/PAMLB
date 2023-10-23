# VQNL (Orig: Natural Language Video Localization)

[Original source code](https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet) of PyTorch implementation of the **VSLNet** baseline for the Ego4D: Natural Language Queries (NLQ) task ([ArXiv](https://arxiv.org/abs/2110.07058), [webpage](https://ego4d-data.org/)).

## Installation instructions
1. Clone the source code.

	```
	git clone https://github.com/escalab/PAMLB.git
	cd VQNL/
	```

2. Create `conda` environment.

	```
	conda create -n vqnl python=3.8
	conda activate vqimg
	```

3. Environment setup

	```
	pip install -r requirements.txt
	```	
	
	Note: Check whether the version of `torch` and `torchvision` match your hardware.
	
## Data preparation

1. Install data downloader.

	```
	python3 -m nltk.downloader punkt
	```
	
2. Download the dataset (`nlq_{train, val, test_unannotated}.json`) from the [official webpage](https://ego4d-data.org/) and place them in `data/` folder.

3. Download the video features released from [official website](https://ego4d-data.org/) and place them in `data/features/nlq_official_v1/video_features/` folder. We used `omnivore_video_swinl_fp16` video features in our experiments.

	Note: The size of v1.0 `omnivore_video_swinl_fp16` video feature is roughly 11`GB`. For the insufficient space on Jetson platform, you can try to use `Docker volume` and create a soft link to video features.

	```
	# create Docker volume
	docker volume create --driver local \ 
	--opt type=nfs \ 
	--opt o=addr=[ip-address] \ 
	--opt device=:[path-to-directory] [volume-name]

	# start the container
	docker run -it --runtime nvidia \
	--network=host \
	--name vqnl \
	--mount source=[volume-name],target=/mnt/trace \
	nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

	ln -s /mnt/trace <PATH-TO-PAMLB>/VQNL/data/features/nlq_official_v1/video_features
	```

## Training and inference

1. Setup `vars.sh` for environment variables.

	```sh
	#!/bin/bash
	export NAME=official_v1
	export TASK_NAME=nlq_$NAME
	export BASE_DIR=data/dataset/nlq_$NAME
	export FEATURE_BASE_DIR=data/features/nlq_$NAME
	export FEATURE_DIR=$FEATURE_BASE_DIR/video_features
	export VQNL_ROOT=<PATH-TO-PAMLB>/VQNL
	export MODEL_BASE_DIR=$VQNL_ROOT/checkpointsnts
	```

2. Run the data preprocessing script: 

	```
	chmod +x ./prepare_dataset.sh
	./prepare_dataset.sh
	```

	This generates JSON files in `data/dataset/nlq_official_v1` and processed clip features in `data/features/nlq_official_v1/official` that can be used for training and evaluating the VSLNet baseline model.


3. Train a model using `run_train.sh` script.
	
	```
	# edit conda path if necessary
	# <PATH-TO-CONDA>/bin/activate vqnl

	chmod +x ./run_train.sh
	./run_train.sh
	```
	It generates VSLNet checkpoints under `checkpoints/$NAME/` folder.
	
4. Predict on test set.

	```
	# edit conda path if necessary
	# <PATH-TO-CONDA>/bin/activate vqnl

	chmod +x ./run_test.sh
	./run_test.sh
	```
	
5. Redirect output and use `vqnl_perf_parser.py` script to parse the performance numbers.

	```
	./run_test.sh > vqnl_infer_perf.txt
	python vqnl_perf_parser.py --file vqnl_infer_perf.txt
	```

## Hardware/software specifications
- X86
	- NVIDIA RTX 3090 GPU
		- CUDA 12.1
		- PyTorch v2.1.0.dev20230501+cu121
		- torchvision v0.16.0.dev20230501+cu121
	- Ubuntu 20.04
- NVIDIA Orin NX Developer Kit (8GB)
	- Jetpack 5.1.1 [L4T 35.3.1]
	- l4t-pytorch:r35.2.1-pth2.0-py3
		- PyTorch v2.0.0
		- torchvision v0.14.1
		- torchaudio v0.13.1

## Acknowledgements
Original codebase from [NLQ](https://github.com/EGO4D/episodic-memory/tree/main/NLQ/VSLNet) and citations of NLQ dataset and baseline implementation.

```
@article{Ego4D2021,
  author={Grauman, Kristen and Westbury, Andrew and Byrne, Eugene and Chavis, Zachary and Furnari, Antonino and Girdhar, Rohit and Hamburger, Jackson and Jiang, Hao and Liu, Miao and Liu, Xingyu and Martin, Miguel and Nagarajan, Tushar and Radosavovic, Ilija and Ramakrishnan, Santhosh Kumar and Ryan, Fiona and Sharma, Jayant and Wray, Michael and Xu, Mengmeng and Xu, Eric Zhongcong and Zhao, Chen and Bansal, Siddhant and Batra, Dhruv and Cartillier, Vincent and Crane, Sean and Do, Tien and Doulaty, Morrie and Erapalli, Akshay and Feichtenhofer, Christoph and Fragomeni, Adriano and Fu, Qichen and Fuegen, Christian and Gebreselasie, Abrham and Gonzalez, Cristina and Hillis, James and Huang, Xuhua and Huang, Yifei and Jia, Wenqi and Khoo, Weslie and Kolar, Jachym and Kottur, Satwik and Kumar, Anurag and Landini, Federico and Li, Chao and Li, Yanghao and Li, Zhenqiang and Mangalam, Karttikeya and Modhugu, Raghava and Munro, Jonathan and Murrell, Tullie and Nishiyasu, Takumi and Price, Will and Puentes, Paola Ruiz and Ramazanova, Merey and Sari, Leda and Somasundaram, Kiran and Southerland, Audrey and Sugano, Yusuke and Tao, Ruijie and Vo, Minh and Wang, Yuchen and Wu, Xindi and Yagi, Takuma and Zhu, Yunyi and Arbelaez, Pablo and Crandall, David and Damen, Dima and Farinella, Giovanni Maria and Ghanem, Bernard and Ithapu, Vamsi Krishna and Jawahar, C. V. and Joo, Hanbyul and Kitani, Kris and Li, Haizhou and Newcombe, Richard and Oliva, Aude and Park, Hyun Soo and Rehg, James M. and Sato, Yoichi and Shi, Jianbo and Shou, Mike Zheng and Torralba, Antonio and Torresani, Lorenzo and Yan, Mingfei and Malik, Jitendra},
  title     = {Ego4D: Around the {W}orld in 3,000 {H}ours of {E}gocentric {V}ideo},
  journal   = {CoRR},
  volume    = {abs/2110.07058},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.07058},
  eprinttype = {arXiv},
  eprint    = {2110.07058}
}
```

```
@inproceedings{zhang2020span,
    title = "Span-based Localizing Network for Natural Language Video Localization",
    author = "Zhang, Hao  and Sun, Aixin  and Jing, Wei  and Zhou, Joey Tianyi",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.585",
    pages = "6543--6554"
}
```

We utilized the NLQ codebase to measure the performance of emerging personal assistant applications and generated insights for different data processing models.