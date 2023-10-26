# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import pickle
import time

import numpy as np
import torch
import tqdm
import sys

from mtr.utils import common_utils

def deepgso(ob):
    size = sys.getsizeof(ob)
    
    if isinstance(ob, (list,tuple,set)):
        for element in ob:
            size+=deepgso(element)
    if isinstance(ob, dict):
        for k,v in ob.items():
            size+=deepgso(k)
            size+=deepgso(v)
    return size

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, logger_iter_interval=50):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    print(f"dataloader len {len(dataloader)}")

    pred_dicts = []

    for i, batch_dict in enumerate(dataloader): # 44097 samples/ 8 batch_size = 5513
        with torch.no_grad():
            s_time = time.time()
            batch_pred_dicts = model(batch_dict)
            batch_inference_time = time.time() - s_time
            # print(f"batch_pred_dicts size {deepgso(batch_pred_dicts)} bytes")

            # project the coordinate from agent space to world space
            s_time = time.time()
            final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, output_path=final_output_dir if save_to_file else None)
            pred_dicts += final_pred_dicts
            post_process_time = time.time() - s_time
            print(f"batch_inference_time {batch_inference_time:.6f}s; batch_postprocess_time {post_process_time:.6f}s")

        disp_dict = {}

        # Due to log_interval = 50, there will be 5513 / 50 = 112 log info
        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(dataloader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(dataloader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None) # Otto: quite independent to these metrics
            logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(dataloader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}, ')
                        # f'inference_time: {inference_time}s')

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        pred_dicts = common_utils.merge_results_dist(pred_dicts, len(dataset), tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    overall = (time.time() - start_time)
    print(f"Overall time: {overall:.6f}s")
    sec_per_example = overall / len(dataloader.dataset)
    print(f"dataloader.dataset len {len(dataloader.dataset)}")
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(pred_dicts, f)

    result_str, result_dict = dataset.evaluation(
        pred_dicts,
        output_path=final_output_dir, 
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict


if __name__ == '__main__':
    pass
