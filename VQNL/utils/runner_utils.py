import copy
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm
from enum import Enum

import utils.evaluate_ego4d_nlq as ego4d_eval
from utils.data_util import index_to_time

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def filter_checkpoints(model_dir, suffix="t7", max_to_keep=5):
    model_paths = glob.glob(os.path.join(model_dir, "*.{}".format(suffix)))
    if len(model_paths) > max_to_keep:
        model_file_dict = dict()
        suffix_len = len(suffix) + 1
        for model_path in model_paths:
            step = int(os.path.basename(model_path).split("_")[1][0:-suffix_len])
            model_file_dict[step] = model_path
        sorted_tuples = sorted(model_file_dict.items())
        unused_tuples = sorted_tuples[0:-max_to_keep]
        for _, model_path in unused_tuples:
            os.remove(model_path)


def get_last_checkpoint(model_dir, suffix="t7"):
    model_filenames = glob.glob(os.path.join(model_dir, "*.{}".format(suffix)))
    model_file_dict = dict()
    suffix_len = len(suffix) + 1
    for model_filename in model_filenames:
        step = int(os.path.basename(model_filename).split("_")[1][0:-suffix_len])
        model_file_dict[step] = model_filename
    sorted_tuples = sorted(model_file_dict.items())
    last_checkpoint = sorted_tuples[-1]
    return last_checkpoint[1]


def convert_length_to_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.size()[0], max_len
    ) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def eval_test(
    model,
    data_loader,
    device,
    mode="test",
    result_save_path=None,
    gt_json_path=None,
    epoch=None,
    global_step=None,
):
    # batch_time = AverageMeter('Batch_time', ':12.3f', Summary.NONE)
    # data_time = AverageMeter('Data_time', ':12.3f')
    # progress = ProgressMeter(
    #     len(data_loader),
    #     # [batch_time, data_time],
    #     # [batch_time],
    #     [data_time],
    #     prefix='Test: ')
    
    predictions = []

    model.eval() # Otto
    # end = time.time()
    with torch.no_grad():
        
        # for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in enumerate(data_loader): # Otto
        for idx, (records, vfeats, vfeat_lens, word_ids, char_ids) in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="evaluate {}".format(mode),
        ):
            # data_time.update(time.time() - end)
            # print(f"data_time {time.time() - end} sec")
            
            # start_time = time.time() # old
            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)

            if isinstance(word_ids, dict):
                word_ids = {key: val.to(device) for key, val in word_ids.items()}
                # generate mask
                query_mask = (
                    (torch.zeros_like(word_ids["input_ids"]) != word_ids["input_ids"])
                    .float()
                    .to(device)
                )
            else:
                word_ids, char_ids = word_ids.to(device), char_ids.to(device)
                # generate mask
                query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)

            # generate mask
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            # feature_prepare_time = time.time() - start_time # old

            start_time = time.time() # old
            # compute predicted results
            _, start_logits, end_logits = model(
                word_ids, char_ids, vfeats, video_mask, query_mask
            )
            start_indices, end_indices = model.extract_index(start_logits, end_logits)
            start_indices = start_indices.cpu().numpy()
            end_indices = end_indices.cpu().numpy()
            prediction_time = time.time() - start_time # old
            # batch_time.update(time.time() - end)
            # end = time.time()
            # print(f"feature prep time {feature_prepare_time} sec")
            print(f"inference time {prediction_time:.6f} sec")
            # Record output and use standard evalution script for NLQ.
            # post_process_start = time.time()
            for record, starts, ends in zip(records, start_indices, end_indices):
                # Convert all indices to times.
                timewindow_predictions = []
                # per_post_process_start = time.time() # old 
                for start, end in zip(starts, ends):
                    start_time, end_time = index_to_time(
                        start, end, record["v_len"], record["duration"]
                    )
                    timewindow_predictions.append([float(start_time), float(end_time)])
                new_datum = {
                    "clip_uid": record["vid"],
                    "annotation_uid": record["annotation_uid"],
                    "query_idx": int(record["query_idx"]),
                    "predicted_times": copy.deepcopy(timewindow_predictions),
                }
                predictions.append(new_datum)
                # print(f"Per-post process time {time.time() - per_post_process_start} sec") # old

            # post_process_end = time.time() - post_process_start # old
            # print(f"Per-batch time {batch_time.val} average time {batch_time.avg}")
            # print(f"Eval time: {eval_et / 60.0:>6.6f} mins")
            # if idx % 5 == 0:

            # progress.display(idx + 1)
        # print(f"Sum of batch time {batch_time.sum} sec")
        # progress.display_summary() is one ☝️ 

    # Save predictions if path is provided.
    if result_save_path:
        with open(result_save_path, "w") as file_id:
            json.dump(
                {
                    "version": "1.0",
                    "challenge": "ego4d_nlq_challenge",
                    "results": predictions,
                }, file_id
            )

    # Evaluate if ground truth JSON file is provided.
    if gt_json_path:
        print(f"ground truth JSON is provided!")
        with open(gt_json_path) as file_id:
            ground_truth = json.load(file_id)
        thresholds = [0.3, 0.5, 0.01]
        topK = [1, 3, 5]
        results, mIoU = ego4d_eval.evaluate_nlq_performance(
            predictions, ground_truth, thresholds, topK
        )
        title = f"Epoch {epoch}, Step {global_step}"
        display_results = ego4d_eval.display_results(
            results, mIoU, thresholds, topK, title=title
        )
    else:
        results = None
        mIoU = None
        display_results = None
    return results, mIoU, display_results