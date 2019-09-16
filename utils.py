import pickle,os
from PIL import Image
import scipy.io
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from pdb import set_trace as st

# other util
def accuracy(outputs, targets, lemniscate = None, trainloader = None, sigma = 0.1, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = targets.size(0)
    K = 200

    features = outputs.cuda()
    targets = targets.cuda()
    trainLabels = torch.LongTensor(trainloader.dataset.values).cuda()
    trainLabels = torch.sub(trainLabels, 1)

    trainFeatures = lemniscate.memory.t()#[:,:trainLabels.shape[0]]
    C = trainLabels.max() + 1
    retrieval_one_hot = torch.zeros(K, C).cuda()
    # st()
    dist = torch.mm(features, trainFeatures)
    yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
    candidates = trainLabels.view(1,-1).expand(batch_size, -1)
    # yi = torch.clamp(yi, 0, candidates.shape[1] - 1 )

    retrieval = torch.gather(candidates, 1, yi)
    retrieval_one_hot.resize_(batch_size * K, C ).zero_()
    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
    yd_transform = yd.clone().div_(sigma).exp_()
    probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1 , C), yd_transform.view(batch_size, -1, 1)), 1)
    _, predictions = probs.sort(1, True)
    st()
    correct = predictions.eq(targets.data.view(-1,1))

    # _, pred = outputs.topk(maxk, 1, True, True)
    # pred = pred.t()
    # correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct.narrow(1,0,k).sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

### add
def accuracy_old(outputs, targets, lemniscate = None, trainloader = None, sigma = 0.1, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    # st()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        # correct_k = correct.narrow(1,0,k).sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)

def record_info(info,filename,mode):

    if mode =='train':

        result = (
              'Time {batch_time} '
              'Data {data_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5}\n'
              'LR {lr}\n'.format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5'],lr=info['lr']))
        print result

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5','lr']

    if mode =='test':
        result = (
              'Time {batch_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5} \n'.format( batch_time=info['Batch Time'],
               loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))
        print result
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Loss','Prec@1','Prec@5']

    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)


