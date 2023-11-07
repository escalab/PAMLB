import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json         # read json
import time
import torch
from TTS.api import TTS
import whisper
import os
import argparse
from fastchat.utils import run_cmd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer # M2M100 is a multilingual encoder-decoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # ['en', 'zh']

timings = {}

def start_timing(stage_name, iteration):
    if stage_name not in timings:
        timings[stage_name] = []
    timings[stage_name].append({'iteration': iteration, 'start_time': time.time()})

def end_timing(stage_name, iteration):
    if stage_name in timings:
        for timing_info in timings[stage_name]:
            # if timing_info['iteration'] == 0:
            #     do something
            if timing_info['iteration'] == iteration:
                timing_info['end_time'] = time.time()
                timing_info['elapsed_time'] = timing_info['end_time'] - timing_info['start_time']
                break

def squad_json_to_dataframe_dev(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'],
                                verbose=1):
    """
    input_file_path: path to the squad dev-v2.0.json file.
    record_path: path to the deepest level in json file; default value is ['data', 'paragraphs', 'qas', 'answers'].
    verbose: 0 to suppress output; default is 1.
    """
    if verbose:
        print("Reading the JSON file")
    f = json.loads(open(input_file_path).read())
   
    if verbose:
        print("Processing...")
    
    ## Parsing different level's in the json file
    js = pd.json_normalize(f , record_path )    # ['data','paragraphs','qas','answers']
    m = pd.json_normalize(f, record_path[:-1] ) # ['data','paragraphs','qas']
    r = pd.json_normalize(f,record_path[:-2])   # ['data','paragraphs']

    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx
    main = m[['id','question','context','answers', 'plausible_answers','is_impossible']].set_index('id').reset_index()
    main['plausible_answers'] = main['plausible_answers'].fillna("").apply(list)

    if verbose:
        print("Shape of the DataFrame is {}".format(main.shape))
        print("Done")
    
    return main

def qabot_chat(question_file):
    """
    Using Fastchat cli to run QABot inference. We use vicuna-7b-v1.3 LM as an example.
    question_file: file contains questions.
    """
    models = [
        "lmsys/vicuna-7b-v1.3",
    ]

    for model_path in models:
        if "model_weights" in model_path and not os.path.exists(
            os.path.expanduser(model_path)
        ):
            continue
        cmd = (
            f"python3 -m fastchat.serve.cli --model-path {model_path} "
            f"--style simple < {question_file}"
        )
        ret = run_cmd(cmd)
        if ret != 0:
            return

def generate_qabot_reference_ans(output_file, squad_df):
    if os.path.exists(output_file):
        os.remove(output_file)

    for _id, q in zip((dev['id']).tolist(), (dev['question']).tolist()):
        with open(output_file, 'a') as f:
            f.write(f"{_id}|{q}\n")

def construct_asr_inputs(ref_answers, qabot_input, tts, N_SAMPLES=3):
    sample = open(qabot_input, 'w')

    with open(ref_answers, 'r') as f:
        count = 0
        for line in f:
            _id, q = line.split('|')
            tts.tts_to_file(text=q, file_path=squad_audio_dir+_id+".wav")
            sample.write(f"{q}")
            count += 1
            if (count >= N_SAMPLES):
                sample.close()
                break
    print("Completed text-to-speech task for SQuAD dataset.")

## SQuAD dataset
squad_file_path = 'data/dev-v2.0.json'
record_path = ['data','paragraphs','qas','answers']
verbose = 1
dev = squad_json_to_dataframe_dev(input_file_path=squad_file_path,record_path=record_path)

ref_answers = "squad_questions_ids.txt"
generate_qabot_reference_ans(ref_answers, dev)

## Convert SQuAD textual questions into audio files
os.system("mkdir -p sample_squad_speech")
squad_audio_dir = "sample_squad_speech/"

## Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

## Init TTS with the target model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA", progress_bar=False).to(device)

## Test only: write out N questions as samples
N_SAMPLES = 3
sample_questions = "sample_squad_questions.txt"
construct_asr_inputs(ref_answers, sample_questions, tts, N_SAMPLES)

## Bilingual Voice Assistant application starts
## 1. ASR
audio_files = os.listdir(squad_audio_dir)
print("Loading ASR model...")
start_timing('asr_model_load', 0)
model = whisper.load_model("base")
end_timing('asr_model_load', 0)

wpm_lst = []
num_iter = 0
for f in audio_files:
    num_iter += 1
    start_time = time.time()
    start_timing('asr_inference', num_iter)
    result = model.transcribe(squad_audio_dir+f)
    end_timing('asr_inference', num_iter)
    elapsed = time.time() - start_time
    if verbose:
        print(result["text"])
    temp = result["text"]
    wpm_lst.append(len(temp.split())/(elapsed/60))

print(f"Average words/minute: {sum(wpm_lst)/len(wpm_lst)}")
print(f"{len(wpm_lst)} audio files are transcribed.")

## 2. QABot
default_qa_output_file = "fastchat_qa_output.txt"
if os.path.exists(default_qa_output_file):
    os.remove(default_qa_output_file)

print("Start QABot...")
qabot_chat(sample_questions) # QABot inference answers were written to 'fastchat_qa_output.txt'
## fastchat_qa_output.txt (PAMLB default name in fastchat instrumentation)

## 3. NMT
## Translate 'en' to 'zh' (Chinese)
print("Loading translation model...")

start_timing('nmt_model_load', 0)
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B") # generalized
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B") # generalized
# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh") # specialized (en-zh pair)
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh") # specialized

end_timing('nmt_model_load', 0)

# tokenizer.src_lang = "en"
translated_file = open("translated_qa.txt", "w")
with open(default_qa_output_file, 'r') as f:
    num_iter = 0
    for q in f:
        num_iter += 1
        start_timing('nmt_encode', num_iter)
        encoded_en = tokenizer(q, return_tensors="pt")
        end_timing('nmt_encode', num_iter)

        start_time = time.time()
        start_timing('nmt_inference', num_iter)
        generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id("zh"))
        # generated_tokens = model.generate(**encoded_en)
        inference_elapsed = time.time() - start_time
        end_timing('nmt_inference', num_iter)
        print(f"NMT inference time {inference_elapsed:.6f}s")

        start_timing('nmt_decode', num_iter)
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        # result = [tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens]
        end_timing('nmt_decode', num_iter)
        if verbose:
            print(f"Translated output: {result}")
        translated_file.write(f"{result}\n")

translated_file.close()

## 4. TTS (text in Chinese to audio in Chinese as well)
multilingual_model="tts_models/multilingual/multi-dataset/bark" # generalized
zh_model="tts_models/zh-CN/baker/tacotron2-DDC-GST" # specialized
start_timing('tts_model_load', 0)
tts = TTS(model_name=zh_model, progress_bar=False).to(device)
end_timing('tts_model_load', 0)

# Run TTS inference
with open('translated_qa.txt', 'r') as f:
    num_iter = 0
    for a in f:
        num_iter += 1
        start_timing('tts_inference', num_iter)
        tts.tts_to_file(text=a, file_path=squad_audio_dir+f"{num_iter}_zh"+".wav")
        end_timing('tts_inference', num_iter)

for stage_name, timing_list in timings.items():
    total_elapsed_time = 0
    num_iterations = len(timing_list)

    for timing_info in timing_list:
        if timing_info['iteration'] == 0:
            print(f"{stage_name}: Elapsed Time = {timing_info['elapsed_time']:.6f} seconds")
        elif timing_info['iteration'] > 0:
            total_elapsed_time += timing_info['elapsed_time']
    
    if num_iterations > 1:
        average_elapsed_time = total_elapsed_time / (num_iterations - 1)
        print(f"{stage_name}: Average Elapsed Time = {average_elapsed_time:.6f} seconds")
