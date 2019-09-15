from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from utils import *
import dataloader

import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse
from pdb import set_trace as st

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from network import *

from spatial_cnn_exampler import Spatial_CNN
from motion_cnn import Motion_CNN


parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-t', default=0.1, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    metavar='M', help='momentum for non-parametric updates')

def main():
    global arg
    arg = parser.parse_args()
    print arg

    rgb_preds='record/spatial/spatial_video_preds.pickle'
    opf_preds = 'record/motion/motion_video_preds.pickle'

    data_loader_spacial = dataloader.spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path='./data/jpegs_256/',
                        ucf_list ='./UCF_list/',
                        ucf_split ='01',
                        )
    data_loader_motion = dataloader.Motion_DataLoader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path='./data/tvl1_flow/',
                        ucf_list='./UCF_list/',
                        ucf_split='01',
                        in_channel=10,
                        )

    train_loader_spacial, test_loader_spacial, test_video_spacial = data_loader_spacial.run()
    train_loader_motion, test_loader_motion, test_video_motion = data_loader_motion.run()

    model_spacial = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader_spacial,
                        test_loader=test_loader_spacial,
                        test_video=test_video_spacial,
                        arg = arg
    )
    model_motion = Motion_CNN(
                        # Data Loader
                        train_loader=train_loader_motion,
                        test_loader=test_loader_motion,
                        # Utility
                        start_epoch=arg.start_epoch,
                        resume=arg.resume,
                        evaluate=arg.evaluate,
                        # Hyper-parameter
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        channel = 10*2,
                        test_video=test_video_motion,
                        )

    model_spacial.build_model()
    model_spacial.resume_and_evaluate()
    model_motion.build_model()
    model_motion.resume_and_evaluate()

    cudnn.benchmark = True
    best_prec1_spacial = 0
    best_prec1_motion = 0
    batch_time = AverageMeter()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()
    end = time.time()
    for epoch in range(arg.start_epoch, arg.epochs):
        model_spacial.epoch = epoch
        model_motion.epoch = epoch

        model_spacial.train_1epoch()
        model_motion.train_1epoch()

        prec1_spacial, val_loss_spacial = model_spacial.validate_1epoch()
        prec1_motion, val_loss_motion = model_motion.validate_1epoch()

        # verify it is best or not
        is_best_spacial = prec1_spacial > best_prec1_spacial
        is_best_motion = prec1_motion > best_prec1_motion

        # step scheduler
        model_spacial.scheduler.step(val_loss_spacial)
        model_motion.scheduler.step(val_loss_motion)

        # store if it is best
        if is_best_spacial:
            best_prec1_spacial = prec1_spacial
            with open(rgb_preds, 'wb') as f:
                pickle.dump(model_spacial.dic_video_level_preds, f)
            f.close()

        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model_spacial.model.state_dict(),
                'best_prec1': best_prec1_spacial,
                'optimizer': model_spacial.optimizer.state_dict(),
            },
            is_best_spacial,
            'record/spatial/checkpoint.pth.tar',
            'record/spatial/model_best.pth.tar'
            )

        if is_best_motion:
            best_prec1_motion = prec1_motion
            with open(opf_preds, 'wb') as f:
                pickle.dump(model_motion.dic_video_level_preds, f)
            f.close()

        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model_motion.model.state_dict(),
                'best_prec1': best_prec1_motion,
                'optimizer' : model_motion.optimizer.state_dict()
            },
            is_best_motion,
            'record/motion/checkpoint.pth.tar',
            'record/motion/model_best.pth.tar'
            )

        rgb = model_spacial.dic_video_level_preds
        opf = model_motion.dic_video_level_preds
        video_level_preds = np.zeros((len(rgb.keys()), arg.low_dim))
        video_level_labels = np.zeros(len(rgb.keys()))
        correct = 0
        ii = 0
        for name in sorted(rgb.keys()):
            r = rgb[name]
            o = opf[name]

            label = int(test_video_spacial[name])-1

            video_level_preds[ii,:] = (r+o)
            video_level_labels[ii] = label
            ii+=1
            if np.argmax(r+o) == (label):
                correct += 1

        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()

        top1,top5 = accuracy_old(video_level_preds, video_level_labels, topk=(1,5))

        batch_time.update(time.time() - end)
        end = time.time()
        top1_acc.update(top1)
        top5_acc.update(top5)
        video_loss = 0


        info = {'Epoch':[epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(video_loss,5)],
                'Prec@1':[round(top1_acc.avg,3)],
                'Prec@5':[round(top5_acc.avg,3)]}
        record_info(info, 'record/fusion_test_x.csv','test')











if __name__ == '__main__':
    main()
