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
from LinearAverage import *



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

    #Prepare DataLoader
    data_loader = dataloader.fusion_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path_spatial='./data/jpegs_256/',
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = arg.lr
    if epoch >= 80:
        lr = arg.lr * (0.1 ** ((epoch-80) // 40))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        nb_classes = arg.low_dim
        H = 512
        D_in = 512 * 2
        self.avgpool = nn.AvgPool2d(7)
        self.fc_custom = nn.Linear(D_in, H)
        self.relu = nn.ReLU()
        self.fc_custom_2 = nn.Linear(H, nb_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_custom(x)
        x = self.relu(x)
        x = self.fc_custom_2(x)
        return x

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
        self.spatial_model = resnet18(pretrained= True, channel=3, end2end = self.end2end).cuda() ### all 101 or all 18
        self.motion_model = resnet18(pretrained= True, channel=self.channel, end2end = self.end2end).cuda()

        self.concat_model = MLP().cuda()

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.lemniscate = LinearAverageWithWeights(
                                self.arg.low_dim,
                                self.ndata,
                                self.arg.nce_t,
                                self.arg.nce_m,
                            )


        self.optimizer = torch.optim.SGD(list(self.spatial_model.parameters())+list(self.motion_model.parameters())+list(self.concat_model.parameters()), self.lr, momentum=0.9)
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
            # self.scheduler.step(val_loss)
            adjust_learning_rate(self.optimizer, self.epoch)
            # save model
            if is_best:
                self.best_prec1 = prec1
                with open('record/fusion_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()

            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.spatial_model.state_dict(),
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
            bg = time.time()
            # measure data loading time
            data_time.update(time.time() - end)

            label = label.cuda()
            index = index.cuda()
            data_var = Variable(data_motion).cuda()
            label_var = Variable(label).cuda()
            # compute output for spatial cnn
            output_spatial = Variable(torch.zeros(len(data_spatial['img1']),512, 7, 7).float()).cuda()
            for j in range(len(data_spatial)):
                if j > 0:
                    break
                key = 'img'+str(j)
                data_key = data_spatial[key]
                data_key_var = Variable(data_key).cuda()
                output_spatial += self.spatial_model(data_key_var)
            # compute output for spatial cnn
            output_motion = self.motion_model(data_var)
            input_next = torch.cat((output_spatial, output_motion), 1)
            # st()
            output = self.concat_model(input_next)

            # compute output
            loss = self.criterion(output, label_var)

            # measure accuracy and record loss
            ### add
            # prec1, prec5 = accuracy_old(output.data, label, topk=(1, 5))
            prec1, prec5 = accuracy(output.data, label, lemniscate = self.lemniscate, trainloader = self.train_loader, sigma = self.arg.nce_t, topk=(1, 5))
            losses.update(loss.data.item(), data_motion.size(0))
            top1.update(prec1.item(), data_motion.size(0))
            top5.update(prec5.item(), data_motion.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print("time for this epoch is ", end - bg)

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/fusion_train_supervised.csv','train')


    def validate_1epoch(self):
        print('FUSIONCNN==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.motion_model.eval()
        self.spatial_model.eval()
        self.dic_video_level_preds = {}
        end = time.time()
        progress = tqdm(self.test_loader)
        with torch.no_grad():
            for i, (keys_spatial, keys_motion, data_spatial, data_motion, label, index) in enumerate(progress):
                if i > 3:
                    break ### skip validation
                label = label.cuda()
                index = index.cuda()
                data_motion_var = Variable(data_motion).cuda()
                data_spatial_var = Variable(data_spatial).cuda()
                label_var = Variable(label).cuda()

                # compute output
                output_spatial = self.spatial_model(data_spatial_var)
                output_motion = self.motion_model(data_motion_var)
                input_next = torch.cat((output_spatial, output_motion), 1)
                output = self.concat_model(input_next)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                #Calculate video level prediction
                preds = output.data.cpu().numpy()

                nb_data = preds.shape[0]
                for j in range(nb_data):
                    videoName = keys_motion[j].split('/',1)[0]
                    if videoName not in self.dic_video_level_preds.keys():
                        self.dic_video_level_preds[videoName] = preds[j,:]
                    else:
                        self.dic_video_level_preds[videoName] += preds[j,:]

            video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()

            info = {'Epoch':[self.epoch],
                    'Batch Time':[round(batch_time.avg,3)],
                    'Loss':[round(video_loss,5)],
                    'Prec@1':[round(video_top1,3)],
                    'Prec@5':[round(video_top5,3)]}
            record_info(info, 'record/rgb_test_supervised.csv','test')
        return video_top1, video_loss

    def frame2_video_level_accuracy(self):
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),101))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):

            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name])-1

            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1
            if np.argmax(preds) == (label):
                correct+=1

        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()

        top1,top5 = accuracy_old(video_level_preds, video_level_labels, topk=(1,5))
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())

        top1 = float(top1.numpy())
        top5 = float(top5.numpy())

        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1,top5,loss.data.cpu().numpy()

if __name__=='__main__':
    main()

