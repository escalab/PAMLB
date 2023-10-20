# Parse VQNL perf result
# Usage: python vqnl_perf_parser.py --file vqnl_infer_perf.txt
import re
import argparse

parser = argparse.ArgumentParser(description='Pass params to run experiments.')
parser.add_argument('--file', '-f', type=str, help='performance.csv to be parsed')

args = parser.parse_args()

with open(args.file, 'r') as f:
    
    data_prep_time, model_loading_time, overall_inference_time = 0.0, 0.0, 0.0
    inference_breakdown = []

    for l in f:
        if l.startswith('data preparation time'):
            data_prep_time = float(re.findall(r"\d+.\d+", l)[0]) # sec
        if l.startswith('model loading time'):
            model_loading_time = float(re.findall(r"\d+.\d+", l)[0]) # sec
        if l.startswith('inference time'):
            inference_breakdown.append(float(re.findall(r"\d+.\d+", l)[0]))
        if l.startswith('Overall inference time'):
            overall_inference_time = float(re.findall(r"\d+.\d+", l)[0]) # sec
    
print(f"data preparation time (sec)\tmodel loading time (sec)\toverall inference time (sec)")
print(f"{data_prep_time:.6f}\t{model_loading_time:.6f}\t{overall_inference_time:.6f}")
