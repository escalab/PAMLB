# Recommender

We leverage MIcrosoft News Dataset (MIND) for the recommendation tasks. Original implementat from PURU BEHL [Source (Kaggle)](https://www.kaggle.com/code/accountstatus/mind-microsoft-news-recommendation-v2). The latest version of recommenders can be found at [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders).

Note: The latest version of recommenders works for Python >=3.9 due to some library dependencies. However, the [L4T container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch) for Jetson platform (e.g., Jetson Orin) we used in the experiments only has native Python 3.8.

## Installation instructions
1. Clone the source code.

	```
	git clone https://github.com/escalab/PAMLB.git
	cd Recommender/
	```

2. Create `conda` environment.

	```
	conda create -n rec python=3.8
	conda activate rec
	```

3. Environment setup

	```sh
	pip install -r requirements.txt
	
	# create a Jupyter kernel
	python -m ipykernel install --user --name <environment_name> --display-name <kernel_name>
	```	
	
## Data preparation

1. Download [MINDsmall_train](https://www.kaggle.com/code/trinhtran0104/mind-recommender-from-scratch/input) dataset and unzip into `mind-news-dataset/` folder. Full dataset can be found at [MIND](https://msnews.github.io/).

	```md
	mind-news-dataset
	  ├── MINDsmall_train
	  │        ├── behaviors.tsv
	  │        ├── entity_embedding.vec
	  │        ├── news.tsv
	  │        └── relation_embedding.vec
	  │
	  └── news.tsv
	           └── news.tsv  
	```
	
## Perform inference
	
1. Set up the dataset path and checkpoint store path in `MIND_news_recommender.ipynb`.

2. Run `MIND_news_recommender.ipynb ` using `<kernel_name>` Jupyte kernel.


## Hardware/software specifications
- X86
	- NVIDIA RTX 3090 GPU
		- CUDA 12.1
		- torch 2.1.0
		- pytorch-lightning 2.0.9.post0


- NVIDIA Orin NX Developer Kit (8GB)
	- Jetpack 5.1.1 [L4T 35.3.1]
	- l4t-pytorch:r35.2.1-pth2.0-py3
		- Python 3.8 
		- torch 2.0.0a0+ec3941ad.nv23.2
		- pytorch-lightning 2.0.9.post0


## Acknowledgements

```
@inproceedings{wu2020mind,
  title={Mind: A large-scale dataset for news recommendation},
  author={Wu, Fangzhao and Qiao, Ying and Chen, Jiun-Hung and Wu, Chuhan and Qi, Tao and Lian, Jianxun and Liu, Danyang and Xie, Xing and Gao, Jianfeng and Wu, Winnie and others},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={3597--3606},
  year={2020}
}
```
