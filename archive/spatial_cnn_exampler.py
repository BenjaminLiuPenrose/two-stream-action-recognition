import numpy as np
import pickle
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

import dataloader
from utils import *
from network import *
from LinearAverage import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
    data_loader = dataloader.spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path='./data/jpegs_256/',
                        ucf_list ='./UCF_list/',
                        ucf_split ='01',
                        )

    train_loader, test_loader, test_video = data_loader.run()
    #Model
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_video=test_video,
                        arg=arg
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

class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video, arg):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video
        self.arg=arg

    def build_model(self):
        # self.ndata = self.train_loader.__len__() * self.batch_size
        self.ndata = len(self.train_loader.dataset.values)
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet18(pretrained= True, nb_classes = self.arg.low_dim, channel=3).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.lemniscate = LinearAverageWithWeights(
                                self.arg.low_dim,
                                self.ndata,
                                self.arg.nce_t,
                                self.arg.nce_m,
                            )
        self.optimizer = torch.optim.SGD(list(self.model.parameters()) + list(self.lemniscate.parameters()), self.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                pretrained_dict = checkpoint['state_dict']
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)

                preoptimizer_dict = checkpoint['optimizer']
                optimizer_dict = self.optimizer.state_dict()
                # preoptimizer_dict = {k: v for k, v in preoptimizer_dict.items() if k in optimizer_dict}
                # optimizer_dict.update(preoptimizer_dict)
                self.optimizer.load_state_dict(optimizer_dict)
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

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
                with open('record/spatial/spatial_video_preds_x.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()

            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/spatial/checkpoint_x.pth.tar','record/spatial/model_best_x.pth.tar')

    def train_1epoch(self):
        print('SPCNN==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data_dict, label, index) in enumerate(progress):


            # measure data loading time
            data_time.update(time.time() - end)

            label = label.cuda(async=True)
            target_var = Variable(label).cuda()
            index_var = Variable(index).cuda()
            self.optimizer.zero_grad()
            # compute output
            feature = Variable(torch.zeros(len(data_dict['img1']),self.arg.low_dim).float()).cuda()
            for j in range(len(data_dict)):
                if j > 0:
                    break
                key = 'img'+str(j)
                data = data_dict[key]
                input_var = Variable(data).cuda()

                feature += self.model(input_var)
            # if i > 200:
            #     st()
            output = self.lemniscate(feature, index_var)

            # st()
            loss = self.criterion(output, index_var)
            # loss = self.criterion(feature, target_var)
            # st()
            # measure accuracy and record loss
            prec1, prec5 = accuracy(feature.data, label, lemniscate = self.lemniscate, trainloader = self.train_loader, sigma = self.arg.nce_t, topk=(1, 5))
            # prec1, prec5 = accuracy_old(feature.data, label, topk = (1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # compute gradient and do SGD step
            # self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')

    def validate_1epoch(self):
        print('SPCNN==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        with torch.no_grad():
            for i, (keys,data,label,index) in enumerate(progress):
                if i > 10:
                    break

                label = label.cuda(async=True)
                data_var = Variable(data, volatile=True).cuda(async=True)
                label_var = Variable(label, volatile=True).cuda(async=True)
                index_var = Variable(index, volatile=True).cuda(async=True)

                # compute output
                output = self.model(data_var)
                # if i > 100:
                #     st()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                #Calculate video level prediction
                preds = output.data.cpu().numpy()
                nb_data = preds.shape[0]
                for j in range(nb_data):
                    videoName = keys[j].split('/',1)[0]
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
            record_info(info, 'record/spatial/rgb_test.csv','test')
        return video_top1, video_loss

    def frame2_video_level_accuracy(self):

        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),self.arg.low_dim))
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

        top1,top5 = accuracy(video_level_preds, video_level_labels, lemniscate = self.lemniscate, trainloader = self.train_loader, topk=(1,5))
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())

        top1 = float(top1.cpu().numpy())
        top5 = float(top5.cpu().numpy())

        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1,top5,loss.data.cpu().numpy()







if __name__=='__main__':
    main()