# Example usage: python vqimg_parser.py --file vqimg_output.txt
import json
import argparse
import re

parser = argparse.ArgumentParser(description='Pass params to run experiments.')
parser.add_argument('--file', '-f', type=str, help='vqimg.txt to be parsed')

args = parser.parse_args()

with open(args.file, 'r') as f:
    search_window, clip_read_time, detection_time, peak_signal_time, tracking_time = [], [], [], [], []
    
    q_num = 0
    for line in f:
        if line.startswith('====>'):
            q_num += 1
            data = line.split('|')
            sw = int(re.findall(r'\d+', data[1])[0])
            crt = float(re.findall(r'\d+.\d+', data[2])[0])
            dt = float(re.findall(r'\d+.\d+', data[3])[0])
            pst = float(re.findall(r'\d+.\d+', data[4])[0])
            tt = float(re.findall(r'\d+.\d+', data[5])[0])

            search_window.append(sw)
            clip_read_time.append(crt)
            detection_time.append(dt)
            peak_signal_time.append(pst)
            tracking_time.append(tt)

print(f"Total queries {q_num}")
print(f"search window (frames)\tclip read time (min)\tdetection time (min)\tpeak signal time (min)\ttracking time (min)")
for i in range(q_num):
    print(f"{search_window[i]}\t{clip_read_time[i]:.6f}\t{detection_time[i]:.6f}\t{peak_signal_time[i]:.6f}\t{tracking_time[i]:.6f}")