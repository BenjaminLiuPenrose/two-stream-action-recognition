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

from spatial_cnn import Spatial_CNN
from motion_cnn import Motion_CNN


parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

def main():
    global arg
    arg = parser.parse_args()
    print arg

    #Prepare DataLoader
    data_loader = dataloader.fusion_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path_spatial='./data/jpegs_256/'
                        path_motion='./data/tvl1_flow/',
                        ucf_list='./UCF_list/',
                        ucf_split='01',
                        in_channel=10,
                        )

    train_loader,test_loader, test_video = data_loader.run()
    #Model
    model = Fusion_CNN(
                        # Data Loader
                        train_loader=train_loader,
                        test_loader=test_loader,
                        # Utility
                        start_epoch=arg.start_epoch,
                        resume=arg.resume,
                        evaluate=arg.evaluate,
                        # Hyper-parameter
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        channel = 10*2,
                        test_video=test_video
                        )
    #Training
    model.run()



class Fusion_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, channel,test_video, end2end = True):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.channel=channel
        self.test_video=test_video
        self.end2end = end2end

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        self.spatial_model = resnet18(pretrained= True, channel=3, end2end = self.end2end).cuda()
        self.motion_model = resnet101(pretrained= True, channel=self.channel, end2end = self.end2end).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(list(self.spatial_model.parameters())+list(self.motion_model.parameters()), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)

    def resume_and_evaluate(self):
        pass

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('record/fusion_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()

            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/checkpoint.pth.tar','record/model_best.pth.tar')

    def train_1epoch(self):
        print('FUSION CNN==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.spatial_model.train()
        self.motion_model.train()
        end = time.time()
        progress = tqdm(self.train_loader)
        ### data_dict from spatial, data from motion
        for i, (data_spatial, data_motion, label, index) in enumerate(progress):
            # measure data loading time
            data_time.update(time.time() - end)

            label = label.cuda()
            index = index.cuda()
            data_var = Variable(data_motion).cuda()
            label_var = Variable(label).cuda()
            # compute output for spatial cnn
            output_spatial = Variable(torch.zeros(len(data_spatial['img1']),101).float()).cuda()
            for j in range(len(data_spatial)):
                if j > 0:
                    break
                key = 'img'+str(j)
                data_key = data_spatial[key]
                data_key_var = Variable(data_key).cuda()
                output_spatial += self.model(data_key_var)
            # compute output for spatial cnn
            output_motion = self.model(data_var)
            st()

if __name__=='__main__':
    main()










if __name__ == '__main__':
    main()
