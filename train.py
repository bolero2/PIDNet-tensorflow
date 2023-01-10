# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import json
import os
import sys
PYTHON_UTILS = os.getenv("PYTHON_UTILS")
sys.path.append(PYTHON_UTILS)
import pprint

import logging
import timeit
from glob import glob
from random import shuffle
import traceback

import numpy as np

import tensorflow as tf
from tensorboardX import SummaryWriter
import models

from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
# from utils.function import train, validate
# from utils.utils import create_logger, FullModel
from models.pidnet import get_seg_model
from datasets.custom import CustomDataset

from datetime import datetime
import pickle

_GPUS = tf.config.experimental.list_physical_devices('GPU')


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()
    jobname = os.path.basename(args.cfg).split('.')[0]
    print("\n\n >>> JOB NAME :", jobname, "\n\n")
    loss_logs = []

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        tf.random.set_seed(int(args.seed))        

    # logger, final_output_dir, tb_log_dir = create_logger(
    #     config, args.cfg, 'train')

    # logger.info(pprint.pformat(args))
    # logger.info(config)

    # writer_dict = {
    #     'writer': SummaryWriter(tb_log_dir),
    #     'train_global_steps': 0,
    #     'valid_global_steps': 0,
    # }

    # cudnn related setting
    # cudnn.benchmark = config.CUDNN.BENCHMARK
    # cudnn.deterministic = config.CUDNN.DETERMINISTIC
    # cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    # if torch.cuda.device_count() != len(gpus):
    #     print("The gpu numbers do not match!")
    #     return 0
    
    model = models.pidnet.get_seg_model(config, pretrained=config.MODEL.PRETRAINED)
    # model.summary()
 
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    reshape_size = (config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])

    if config.DATASET.PICKLE_NAME != "":
        pass
        # print("Pre-saved Pickle files will be loaded...")
        # preload_path = os.path.join("data", "pickles", config.DATASET.PICKLE_NAME)
        # assert os.path.isfile(os.path.join(preload_path, 'train.pkl')), "Wrong dataset pickle file path!"

        # trainpack = DatasetParser.load_dataset(os.path.join(preload_path, "train.pkl"))
        # validpack = DatasetParser.load_dataset(os.path.join(preload_path, "valid.pkl"))
        # testpack = DatasetParser.load_dataset(os.path.join(preload_path, "test.pkl"))

        # # Real Training for last...
        # trainpack = trainpack + testpack

    else:
        trainpack = open(os.path.join(config.DATASET.ROOT, config.DATASET.TRAIN_SET), 'r').readlines()
        trainpack = [x.replace('\n', '') for x in trainpack]
        
        validpack = open(os.path.join(config.DATASET.ROOT, config.DATASET.VALID_SET), 'r').readlines()
        validpack = [x.replace('\n', '') for x in validpack]
        
        # (dc) dataset state save pickle format
        pklpath = os.path.join("data", "pickles", jobname) 
        if not os.path.isdir(pklpath):
            os.makedirs(pklpath, exist_ok=True)
        print("\n >>> Dataset pickle will be made ->", pklpath)

        train_pkl_path = os.path.join(pklpath, "train.pkl")
        valid_pkl_path = os.path.join(pklpath, "valid.pkl")

        with open(train_pkl_path, "wb") as pkl_file:
            pickle.dump(trainpack, pkl_file)
        with open(valid_pkl_path, "wb") as pkl_file:
            pickle.dump(validpack, pkl_file)

    # model = get_seg_model(cfg=config, pretrained=config.MODEL.PRETRAINED)
    start_epoch = int(config.TRAIN.BEGIN_EPOCH)
    end_epoch = int(config.TRAIN.END_EPOCH)
    batch_size = int(config.TRAIN.BATCH_SIZE_PER_GPU)

    img_size = config.TRAIN.IMAGE_SIZE

    trainloader = CustomDataset(config=config, datapack=trainpack,
                                batch_size=batch_size, shuffle=True,
                                multi_scale=config.TRAIN.MULTI_SCALE,
                                flip=config.TRAIN.FLIP,
                                ignore_label=config.TRAIN.IGNORE_LABEL,
                                base_size=config.TRAIN.BASE_SIZE,
                                crop_size=crop_size,
                                reshape_size=reshape_size,
                                scale_factor=config.TRAIN.SCALE_FACTOR)

    valid_crop_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    valid_reshape_size = (config.TEST.IMAGE_SIZE[0], config.TEST.IMAGE_SIZE[1])

    validloader = CustomDataset(config=config, datapack=validpack,
                                batch_size=batch_size, shuffle=True,
                                multi_scale=False,
                                flip=False,
                                ignore_label=config.TRAIN.IGNORE_LABEL,
                                base_size=config.TRAIN.BASE_SIZE,
                                crop_size=valid_crop_size,
                                reshape_size=valid_reshape_size)

    # if config.TRAIN.OPTIMIZER.lower() == 'sgd':

    for epoch in range(start_epoch, end_epoch):
        for step, data in enumerate(trainloader):
            print(data)



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Exception :", e)
        traceback.print_exc()
    print("End.")
