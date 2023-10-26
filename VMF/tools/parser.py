# Usage: python parser.py --file result.txt
import argparse
import re

parser = argparse.ArgumentParser(description='Pass params to parse result numbers.')
parser.add_argument('--file', '-f', type=str, help=' ')

args = parser.parse_args()

with open(args.file, 'r') as f:
    iter_cost, batch_inference_time, batch_postprocess_time = 0.0, 0.0, 0.0
    overall = 0.0 # total inference+post-processing time
    cnt = 0
    for l in f:
        if "iter_cost=" in l:
            temp = l.split(',')
            iter_c = float(re.findall(r"\d+.\d+", temp[-4])[0]) # sec
            # print(f"iter_cost {iter_c}s")
            iter_cost += iter_c
        if l.startswith('batch_inference_time'):
            cnt += 1
            bit = float(re.findall(r"\d+.\d+", l)[0]) # sec
            bpt = float(re.findall(r"\d+.\d+", l)[1]) # sec
            batch_inference_time += bit
            batch_postprocess_time += bpt
        if l.startswith('Overall time'):
            overall = float(re.findall(r"\d+.\d+", l)[0]) # sec

    print(f"len(dataloader.dataset) {cnt}")
    print(f"Total iter_cost: {iter_cost:.6f}s")
    print(f"Overall time {overall:.6f}s")
    print(f"Total batch_inference_time {batch_inference_time:.6f}s, total batch_postprocess_time {batch_postprocess_time:.6f}s")
    print(f"Avg. batch_inference_time {(batch_inference_time/cnt):.6f}s\t Avg. batch_postprocess_time {(batch_postprocess_time/cnt):.6f}s")
