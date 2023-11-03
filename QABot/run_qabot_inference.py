"""QABot inference using Vicuna-7b-v1.3 on SQuAD dev-v2.0 dataset."""
import argparse
import os
from fastchat.utils import run_cmd

parser = argparse.ArgumentParser(description='Pass input question file.')
parser.add_argument('--file', '-f', type=str, help='squad_questions.txt')

args = parser.parse_args()

def run_single_gpu():
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
            f"--style simple < {args.file}"
        )
        ret = run_cmd(cmd)
        if ret != 0:
            return

if __name__ == "__main__":
    run_single_gpu()
