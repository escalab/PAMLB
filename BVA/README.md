# Bilingual Voice Assistant (BVA)

We synthesize the complex bilingual voice assistant use case to demonstrate the flexibility/extensibility of our proposed device-agnostic query language (DAQL).

BVF consists of 4 components, including automatic speech recognition (ASR), QABot, neural machine translation (NMT), and text-to-speech (TTS). We leverage [Whisper](https://github.com/openai/whisper/tree/main) for ASR tasks, [FastChat](https://github.com/lm-sys/FastChat/tree/main) for QABot tasks, [M2M100 1.2B](https://huggingface.co/facebook/m2m100_1.2B) and [opus-mt-en-zh](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) for NMT tasks, and [TTS](https://github.com/coqui-ai/TTS) library for text-to-speech generation task. 

## Installation instructions
1. Clone the source code.

	```
	git clone https://github.com/escalab/PAMLB.git
	cd BVA/
	```

2. Create `conda` environment.

	```
	conda create -n bva python=3.9
	conda activate bva
	```
	
	Note: TTS is tested on Ubuntu 18.04 with python >= 3.9, < 3.12. 

3. Environment setup

	```sh
	pip install -r requirements.txt
	
	# on Ubuntu
	sudo apt update && sudo apt install ffmpeg
	```	
	
## Data preparation

1. Download [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset and place them into `data/` folder.

	```md
	data
	  ├── train-v2.0.json
	  └── dev-v2.0.json 	    
	```
	
## Perform inference
	
1. Run voice assistant application using `run_voice_assistant.py` script.

	```
	python run_voice_assistant.py
	```

2. Parse BVA results using `bva_results_parser.py` script.

	```
	python run_voice_assistant.py > bva_results.txt
	
	python bva_results_parser.py --file bva_results.txt
	```

## Hardware/software specifications
- X86
	- NVIDIA RTX 3090 GPU
		- CUDA 12.1
		- PyTorch 2.0.1
	- Ubuntu 20.04
	- QABot -- vicuna-7b-v1.3
	- NMT   -- facebook/m2m100_1.2B
	- TTS   -- tts_models/zh-CN/baker/tacotron2-DDC-GST


Note: Jetson platform could not execute `run_voice_assistant.py` end-to-end due to QABot (device memory constraint), we measure the performance componet by component (i.e., ASR, NMT, and TTS) for ODML data management model.

## Acknowledgements

```
@misc{radford2022robust,
      title={Robust Speech Recognition via Large-Scale Weak Supervision}, 
      author={Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine McLeavey and Ilya Sutskever},
      year={2022},
      eprint={2212.04356},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}

@misc{fan2020englishcentric,
      title={Beyond English-Centric Multilingual Machine Translation}, 
      author={Angela Fan and Shruti Bhosale and Holger Schwenk and Zhiyi Ma and Ahmed El-Kishky and Siddharth Goyal and Mandeep Baines and Onur Celebi and Guillaume Wenzek and Vishrav Chaudhary and Naman Goyal and Tom Birch and Vitaliy Liptchinsky and Sergey Edunov and Edouard Grave and Michael Auli and Armand Joulin},
      year={2020},
      eprint={2010.11125},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@InProceedings{TiedemannThottingal:EAMT2020,
  author = {J{\"o}rg Tiedemann and Santhosh Thottingal},
  title = {{OPUS-MT} — {B}uilding open translation services for the {W}orld},
  booktitle = {Proceedings of the 22nd Annual Conferenec of the European Association for Machine Translation (EAMT)},
  year = {2020},
  address = {Lisbon, Portugal}
 }

@misc{CoquiTTS,
    title = {Coqui TTS},
    abstract = {A deep learning toolkit for Text-to-Speech, battle-tested in research and production},
    year = {2021},
    author = {Gölge Eren and The Coqui TTS Team},
    version = {1.4},
    doi = {10.5281/zenodo.6334862},
    license = {MPL-2.0},
    url = {https://www.coqui.ai},
    keywords = {machine learning, deep learning, artificial intelligence, text to speech, TTS},
    howpublished = {\url{https://github.com/coqui-ai/TTS}}
}

```
