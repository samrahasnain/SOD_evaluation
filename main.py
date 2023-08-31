import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluator import Eval_thread
from dataloader import EvalDataset
# from concurrent.futures import ThreadPoolExecutor
def main(cfg):
    
    if cfg.save_dir is not None:
        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)
            output_dir = cfg.save_dir
        else:
            output_dir = cfg.save_dir
        

    gt_dir = cfg.gt_dir
    pred_dir = cfg.pred_dir

    
    
    loader = EvalDataset(pred_dir, gt_dir)

    thread = Eval_thread(loader, cfg.dataset, output_dir, cfg.cuda)
            
    print(thread.run())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--gt_dir', type=str, default='./gt')
    parser.add_argument('--pred_dir', type=str, default='./pred_maps')
    parser.add_argument('--save_dir', type=str, default='./score')
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)
