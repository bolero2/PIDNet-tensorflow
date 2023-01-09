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
from utils.function import train, validate
from utils.utils import create_logger, FullModel

from datetime import datetime
import pickle
from bdataset import DatasetParser

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
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0
    
    imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.pidnet.get_seg_model(config, imgnet_pretrained=imgnet)
 
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    reshape_size = (config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])

    # (DC) mOdIfIeD HeRe! ==============================================

    if config.DATASET.PICKLE_NAME != "":
        print("Pre-saved Pickle files will be loaded...")
        preload_path = os.path.join("data", "pickles", config.DATASET.PICKLE_NAME)
        assert os.path.isfile(os.path.join(preload_path, 'train.pkl')), "Wrong dataset pickle file path!"

        trainpack = DatasetParser.load_dataset(os.path.join(preload_path, "train.pkl"))
        validpack = DatasetParser.load_dataset(os.path.join(preload_path, "valid.pkl"))
        testpack = DatasetParser.load_dataset(os.path.join(preload_path, "test.pkl"))

        # Real Training for last...
        trainpack = trainpack + testpack

    else:
        imageset = sorted(glob(os.path.join(config.DATASET.ROOT, config.DATASET.IMAGES, "*.jpg")))
        annotset = sorted(glob(os.path.join(config.DATASET.ROOT, config.DATASET.ANNOTATIONS, "*.png")))
        datapack = []

        for i, a in zip(imageset, annotset):
            datapack.append({i:a})
        shuffle(datapack)
        trainpack = datapack[:int(len(datapack) * 0.8)]
        validpack = datapack[int(len(datapack) * 0.8):int(len(datapack) * 0.9)]
        testpack = datapack[int(len(datapack) * 0.9):]

        # (dc) dataset state save pickle format
        pklpath = os.path.join("data", "pickles", jobname) 
        if not os.path.isdir(pklpath):
            os.makedirs(pklpath, exist_ok=True)
        print("\n >>> Dataset pickle will be made ->", pklpath)

        train_pkl_path = os.path.join(pklpath, "train.pkl")
        valid_pkl_path = os.path.join(pklpath, "valid.pkl")
        test_pkl_path = os.path.join(pklpath, "test.pkl")

        with open(train_pkl_path, "wb") as pkl_file:
            pickle.dump(trainpack, pkl_file)
        with open(valid_pkl_path, "wb") as pkl_file:
            pickle.dump(validpack, pkl_file)
        with open(test_pkl_path, "wb") as pkl_file:
            pickle.dump(testpack, pkl_file)


    train_dataset = eval('datasets.' + config.DATASET.DATASET)(
        datapack=trainpack,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        reshape_size=reshape_size,
        scale_factor=config.TRAIN.SCALE_FACTOR,
        mode='train'
    )

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_reshape_size = (config.TEST.IMAGE_SIZE[0], config.TEST.IMAGE_SIZE[1])

    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        datapack=validpack,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TRAIN.BASE_SIZE,
        crop_size=test_size,
        reshape_size=test_reshape_size,
        mode='valid'
    )
    # =====================================================================
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        drop_last=True,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        # weight=train_dataset.class_weights
                                        )
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    # weight=train_dataset.class_weights
                                    )

    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    elif config.TRAIN.OPTIMIZER == 'adam':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.Adam(params,
                                    lr=config.TRAIN.LR,
                                    weight_decay=config.TRAIN.WD,
                                    betas=(config.TRAIN.MOMENTUM, 0.999))

    elif config.TRAIN.OPTIMIZER == 'adamw':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params,
                                    lr=config.TRAIN.LR,
                                    weight_decay=config.TRAIN.WD,
                                    betas=(config.TRAIN.MOMENTUM, 0.999))
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
        
    best_mIoU = 0
    min_val_loss = np.inf
    last_epoch = 0
    flag_rm = config.TRAIN.RESUME

    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    
    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        # """
        train_loss = train(config, epoch, config.TRAIN.END_EPOCH, 
                          epoch_iters, config.TRAIN.LR, num_iters,
                          trainloader, optimizer, model, writer_dict)

        # if flag_rm == 1 or (epoch % 5 == 0 and epoch < real_end - 100) or (epoch >= real_end - 100):
        valid_loss, mean_IoU, IoU_array = validate(config, 
                    testloader, model, writer_dict)
        if flag_rm == 1:
            flag_rm = 0

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + 'checkpoint.pth.tar'))
        torch.save({
            'epoch': epoch+1,
            'best_mIoU': best_mIoU,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir,'checkpoint.pth.tar'))

        if min_val_loss > valid_loss and mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            min_val_loss = valid_loss
            torch.save(model.module.state_dict(),
                    os.path.join(final_output_dir, 'best.pt'))

            print(f"\n\n -----> Best.pt is saved in {os.path.join(final_output_dir, 'best.pt')}\n")

        msg = 'Validation Loss: {:.3f}, (Valid) MeanIoU: {: 4.4f}, (Valid) Best_mIoU: {: 4.4f}'.format(
                    valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        logging.info(IoU_array)

        loss_logs.append({
            "epoch": epoch,
            "train_loss": np.round(train_loss, 3),
            "valid_loss": np.round(valid_loss, 3)
        })

    torch.save(model.module.state_dict(),
            os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()

    with open(os.path.join("output", config.DATASET.DATASET, jobname, f"{jobname}_losses.json"), 'w') as jsonfile:
        json.dump(loss_logs, jsonfile, indent=4)

    logger.info('Hours: %d' % np.int((end-start)/3600))
    logger.info('Done')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Exception :", e)
        traceback.print_exc()
    print("End.")
