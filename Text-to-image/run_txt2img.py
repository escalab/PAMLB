# Usage: python run_txt2img.py -o $PWD -d <PATH-TO-CUB-200-2011>/text_c10
import os
from diffusers import StableDiffusionPipeline
import torch
import time
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='Pass params to run experiments.')
parser.add_argument('--outdir', '-o', type=str, help='absolute path to store generated images')
parser.add_argument('--dataset', '-d', type=str, help='absolute ath to CUB-200-2011 text dataset')

args = parser.parse_args()

def process(filename, pipe, dir_generated, tot_infer_times):
    with open(filename, 'r') as f:
        counter = 1
        inference_times = 0.0
        text_descriptions = [line.rstrip() for line in f]
        for prompt in text_descriptions:
            if debug_mode:
                print(f"input text: {prompt}")

            start_time = time.time()
            image = pipe(prompt).images[0] # inference (generate image)
            infer_time = time.time() - start_time
            inference_times += infer_time

            image.save(f"{dir_generated}/{counter}.png")
            counter += 1
            
            if debug_mode:
                print(f"Inference time {infer_time:.6f}s/image")
        print(f"Avg. inference time {(inference_times/counter):.6f}s/image per directory")
        tot_infer_times.append(inference_times/counter)

if __name__ == '__main__':
    # path to CUB-200-2011 text dataset
    # cub200_text_dir='<PATH-TO-CUB-200-2011-DATASET/text_c10>'
    cub200_text_dir = args.dataset

    subdir_list = [x[0] for x in os.walk(cub200_text_dir)]
    subdir_list = subdir_list[1:] # skip the current dir

    NUMBER_DIR = 1 # change this number (e.g., 1-200) to generate more results
    # each subdir contains a few text files, and each text file has 10 captions
    subdirs = subdir_list[0:NUMBER_DIR]
    debug_mode = False
    text_files, filenames = [], []

    for subdir in subdirs:
        for f in os.listdir(subdir):
            if f.endswith('.txt'):
                text_files.append(f)
                filename = subdir + '/' + str(f)
                filenames.append(filename)
    
    # Path to store generated images
    # DIR_PREFIX = '<PATH-TO-PAMLB/Text-to-image>'
    DIR_PREFIX = args.outdir

    pretrained_model = "CompVis/stable-diffusion-v1-4"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using {pretrained_model} pretrained model with device: {device}.")
    
    NUMBER_TXT_FILES = 3 # generate NUMBER_TXT_FILES*10 images in total
    tot_infer_times = []
    text_files, filenames = text_files[:NUMBER_TXT_FILES], filenames[:NUMBER_TXT_FILES] # test-only, comment this out to generate complete results 

    start_time = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16)
    pipe.to(device)
    print(f"Model loading time {(time.time() - start_time):.6f}s")

    for dirname, inp in zip(text_files, filenames):
        DIR_GENERATED = DIR_PREFIX + '/' + dirname[:-4] + '_ImgDir'
        Path(DIR_GENERATED).mkdir(parents=True, exist_ok=True)
        process(inp, pipe, DIR_GENERATED, tot_infer_times)

    print("All text-to-image generation tasks are completed!")
    print(f"Overall average inference time {(sum(tot_infer_times)/len(tot_infer_times)):.6f}s")