# Question Answering Bot (QABot)

[Reference - lm-sys/FastChat](https://github.com/lm-sys/FastChat/tree/main). We leverage FastChat implementation for QABot application and instrument timestamps for the evaluation. 

## Installation instructions
1. Clone the source code.

	```
	git clone https://github.com/escalab/PAMLB.git
	cd QABot/
	```

2. Create `conda` environment.

	```
	conda create -n qabot python=3.9
	conda activate qabot
	```

3. Environment setup

	```
	pip install -r requirements.txt
	```	
	
## Data preparation

1. Download [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset and place into `data/` folder.

	```md
	data
	  ├── train-v2.0.json
	  └── dev-v2.0.json 	    
	```

2. Preprocess the SQuAD dataset using `process_squad_dataset.py` script and `squad_questions.txt` will be generated.

	```
	python process_squad_dataset.py
	```
	
## Perform inference
	
1. Run inference using `run_qabot_inference.py` script.

	```
	python run_qabot_inference.py --file squad_questions.txt
	```
	
	Note: `dev-v2.0.json` conatins 11873 questions, you can have a quick test using: 
	
	```
	head -10 squad_questions.txt > partial_squad_questions.txt
	
	python run_qabot_inference.py --file partial_squad_questions.txt
	```

2. Parse QABot results.

	```
	python run_qabot_inference.py --file squad_questions.txt > qa_results.txt
	
	python qabot_results_parser.py --file qa_results.txt
	```

### Not Enough Memory
FastChat [tutorial](https://github.com/lm-sys/FastChat/tree/main#not-enough-memory)

- 8-bit compression
- Offload to CPU
- [GPTQ 4-bit inference](https://github.com/lm-sys/FastChat/blob/main/docs/gptq.md)
	

## Hardware/software specifications
- X86
	- NVIDIA RTX 3090 GPU
		- CUDA 12.1
		- PyTorch 2.1.0
	- Ubuntu 20.04
	- LLM -- vicuna-7b-v1.3


## Acknowledgements

```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena},
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

We utilized FastChat Command Line Interface to measure the performance of LLM inference and provided insights for different data processing models.
