# Text-to-image generation

[Reference library - Diffusers](https://github.com/huggingface/diffusers). A go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules.

## Installation instructions
1. Clone the source code.

	```
	git clone https://github.com/escalab/PAMLB.git
	cd Text-to-image/
	```

2. Create `conda` environment.

	```
	conda create -n txt2img python=3.8
	conda activate txt2img
	```

3. Environment setup

	```
	pip install -r requirements.txt
	```	
	
	
## Data preparation

1. Download Caltech-UCSD Birds-200-2011 ([CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)) dataset. We used the annotation part of the dataset. Below is the structure of the CUB-200-2011 dataset:

	```md
cvpr2016_cub
├── text_c10
│    ├── 001.Black_footed_Albatross
│    │     ├── Black_Footed_Albatross_0001_796111.txt
│    │     ├── ...
│    │     └── Black_Footed_Albatross_0090_796077.txt
│    │
│    ├── 002.Laysan_Albatross
│    .     ├── Laysan_Albatross_0001_545.txt
│    .     ├── ...
│    .     └── Laysan_Albatross_0104_630.txt
│    
│    │   
│    └── 200.Common_Yellowthroat
│			 ├── ...
│          └── ...
│
├── w2v_c10
│   ├── ...
│   └── ...
│   
└── word_c10 
		├── ...
		└── ...  
	```

	Note: We used the text description under `text_c10` folder as inputs for the stable diffusion model.
	
## Perform inference
	
1. Generate images from CUB 101 dataset using `run_txt2img.py` script.

	```
	python run_txt2img.py -o <PATH-TO-STORE-GENERATED-IMAGES> -d <PATH-TO-CUB-200-2011>/text_c10
	```
	

## Hardware/software specifications
- X86
	- NVIDIA RTX 3090 GPU
		- CUDA 12.1
		- PyTorch 2.0.1
		- torchvision 0.15.2
		- diffusers 0.15.0
	- Ubuntu 20.04


## Acknowledgements

```
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

```
@techreport{WahCUB_200_2011,
  title={The caltech-ucsd birds-200-2011 dataset},
  author={Wah, Catherine and Branson, Steve and Welinder, Peter and Perona, Pietro and Belongie, Serge},
  year={2011},
  institution = {California Institute of Technology},
  number = {CNS-TR-2011-001}
}
```

We utilized `diffusers` library to measure the performance of generating images and provided insights for different data processing models.