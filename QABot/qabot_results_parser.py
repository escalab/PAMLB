import re
import argparse

parser = argparse.ArgumentParser(description='Pass params to parse QABot results.')
parser.add_argument('--file', '-f', type=str, help='qabot_results.txt')

args = parser.parse_args()
qabot_pattern = r'\[PAMLBench\] (\w+ [a-zA-Z\s]+) (\d+\.\d+s)'
qabot_timing_data = {
    'Tokenizer loading time': 0.0,
    'Model loading time': 0.0,
    'Prompt encoding time': [],
    'Inference time': [],
    'Decoding time': [],
}

other_pattern = r'(\w+): (Elapsed Time|Average Elapsed Time) = (\d+\.\d+) seconds'
other_timing_data = {}

with open(args.file, 'r') as f:
    
    for l in f:
        matches = re.findall(qabot_pattern, l)
        other_matches = re.findall(other_pattern, l)
        if matches:
            operation, t = matches[0]
            if (operation == 'Tokenizer loading time'):
                qabot_timing_data['Tokenizer loading time'] = float(t[:-1])
            elif (operation == 'Model loading time'):
                qabot_timing_data['Model loading time'] = float(t[:-1])
            elif operation in qabot_timing_data:
                qabot_timing_data[operation].append(float(t[:-1]))
            else:
                continue

        for om in other_matches:
            stage, timing_type, time_value = om
            if stage not in other_timing_data:
                other_timing_data[stage] = {}
            other_timing_data[stage][timing_type] = float(time_value)

for operation, times in qabot_timing_data.items():
    if isinstance(times, float):
        print(f"QABot {operation.lower()}: {times:.6f} seconds")
    elif len(times) > 1:
        average_time = sum(times) / len(times)
        print(f"QABot average {operation.lower()}: {average_time:.6f} seconds")
    else:
        print(f"Timing data parsing error.")

for stage, data in other_timing_data.items():
    if 'Elapsed Time' in data:
        print(f"{stage} time: {data['Elapsed Time']:.6f} seconds")
    if 'Average Elapsed Time' in data:
        print(f"{stage} average time: {data['Average Elapsed Time']:.6f} seconds")
