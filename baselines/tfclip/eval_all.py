import os
import os.path as osp
import sys
import datetime

import scipy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import argparse
from config import cfg

from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader,make_eval_all_dataloader,make_eval_rrs_dataloader
from model.make_model_clipreid import make_model
from loss.make_loss import make_loss
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2, do_inference_dense, do_inference_rrs, do_inference_rrs_visualize


def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

	#############################################
	# --> 加载参数和初始化
	#############################################
	parser = argparse.ArgumentParser(description="ReID Baseline Training")
	parser.add_argument(
		"--config_file", default="configs/vit_clipreid.yml", help="path to config file", type=str
	)
	parser.add_argument("--sort", default="ascending", type=str, help="Sorting order for visualization")
	parser.add_argument("--num_vis", default=10, type=int, help="Number of images to visualize")
	parser.add_argument("--rank_vis", default=10, type=int, help="Number of ranks to visualize")
	parser.add_argument("--vis_output_dir", default="./vis_results", type=str, help="Output directory for visualization results")
	parser.add_argument("--local_rank", default=0, type=int)
	parser.add_argument("opts", help="Modify config options using the command-line", default=None,
	                    nargs=argparse.REMAINDER)

	args = parser.parse_args()
	
	if args.config_file != "":
		cfg.merge_from_file(args.config_file)

	cfg.merge_from_list(args.opts)
	cfg.freeze()
		

	set_seed(cfg.SOLVER.SEED)

	if cfg.MODEL.DIST_TRAIN:
		torch.cuda.set_device(args.local_rank)

	output_dir = cfg.OUTPUT_DIR
	if output_dir and not os.path.exists(output_dir):
		os.makedirs(output_dir)

	logger = setup_logger("transreid", output_dir, if_train=True)
	logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
	logger.info(args)

	if args.config_file != "":
		logger.info("Loaded configuration file {}".format(args.config_file))
		with open(args.config_file, 'r') as cf:
			config_str = "\n" + cf.read()
			logger.info(config_str)
	logger.info("Running with config:\n{}".format(cfg))

	if cfg.MODEL.DIST_TRAIN:
		torch.distributed.init_process_group(backend='nccl', init_method='env://')

	#############################################
	# --> 数据加载
	#############################################
	# val_loader, num_query, num_classes, camera_num, view_num = make_eval_all_dataloader(
	# 	cfg)
	val_loader, num_query, num_classes, camera_num, view_num = make_eval_rrs_dataloader(
		cfg		
	)
	
	model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
	#
	model.load_param(cfg.TEST.WEIGHT)
	# do_inference_dense(cfg, model, val_loader, num_query)
	# do_inference_rrs(cfg, model, val_loader, num_query)
	do_inference_rrs_visualize(
		cfg, 
		model, 
		val_loader, 
		num_query,
		visualize=True,
		sort=args.sort,
		num_vis=args.num_vis,
		rank_vis=args.rank_vis,
		output_dir=args.vis_output_dir
	)
