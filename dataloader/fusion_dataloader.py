import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from split_train_test_video import *
from skimage import io, color, exposure
from pdb import set_trace as st

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau


class fusion_dataset(Dataset):
    def __init__(self, dic_spatial, dic_motion, in_channel, root_dir_spatial,root_dir_motion, mode, transform_spatial=None, transform_motion=None):
        ### args from spatial
        self.keys_spatial = dic_spatial.keys()
        self.values_spatial=dic_spatial.values()
        self.keys_motion=dic_motion.keys()
        self.values_motion=dic_motion.values()
        self.root_dir_spatial = root_dir_spatial
        self.mode =mode ### train or validation
        self.transform_spatial=transform_spatial
        ### args from motion
        self.root_dir_motion = root_dir_motion
        self.in_channel = in_channel
        self.img_rows = 224
        self.img_cols = 224
        self.transform_motion = transform_motion


    def __len__(self):
        return len(self.keys_spatial)

    def stackopf(self):
        name = 'v_'+self.video
        u = self.root_dir_motion+ 'u/' + name
        v = self.root_dir_motion+ 'v/'+ name

        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)


        for j in range(self.in_channel):
            idx = i + j
            idx = str(idx)
            frame_idx = 'frame'+ idx.zfill(6)
            h_image = u +'/' + frame_idx +'.jpg'
            v_image = v +'/' + frame_idx +'.jpg'

            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.transform_motion(imgH)
            V = self.transform_motion(imgV)


            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V
            imgH.close()
            imgV.close()
        return flow

    def load_ucf_image(self,video_name, index):
        if video_name.split('_')[0] == 'HandstandPushups':
            n,g = video_name.split('_',1)
            name = 'HandStandPushups_'+g
            path = self.root_dir_spatial + 'v_'+video_name
        else:
            # path = self.root_dir + video_name.split('_')[0]+'/separated_images/v_'+video_name+'/v_'+video_name+'_'
            path = self.root_dir_spatial + 'v_'+video_name

        img = Image.open(path + "/frame%06d" % index +'.jpg')
        transformed_img = self.transform_spatial(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):
        if self.mode == 'train':
            ### prepare for train spatial dataloader
            video_name, nb_clips = self.keys_spatial[idx].split(' ')
            nb_clips = int(nb_clips)
            clips = []
            clips.append(random.randint(1, nb_clips/3))
            clips.append(random.randint(nb_clips/3, nb_clips*2/3))
            clips.append(random.randint(nb_clips*2/3, nb_clips+1))
            ### prepare for train motion dataloader
            self.video, nb_clips = self.keys_motion[idx].split(' ')
            self.clips_idx = random.randint(1,int(nb_clips))
        elif self.mode == 'val':
            ### prepare for vol spatial dataloader
            video_name, index = self.keys_spatial[idx].split(' ')
            index =abs(int(index))
            ### prepare for val motion dataloader
            self.video,self.clips_idx = self.keys_motion[idx].split(' ')
        else:
            raise ValueError('There are only train and val mode')

        label = self.values_spatial[idx] ### values spatial is the same as values_motion
        label = int(label)-1
        ### data for motion
        data_motion = self.stackopf()

        if self.mode=='train':
            data_spatial ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data_spatial[key] = self.load_ucf_image(video_name, index)

            sample = (data_spatial, data_motion, label, idx)
        elif self.mode=='val':
            data_spatial = self.load_ucf_image(video_name,index)
            sample = (video_name, self.video, data_spatial, data_motion, label, idx)
        else:
            raise ValueError('There are only train and val mode')

        return sample

class fusion_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel, path_spatial, path_motion, ucf_list, ucf_split):
        ### args for spatial dataloader
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.frame_count ={}
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video()
        self.data_path_spatial=path_spatial
        ### args for motion dataloader
        self.data_path_motion =path_motion
        self.in_channel = in_channel

    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        with open('./dataloader/dic/frame_count.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame :
            videoname = line.split('_',1)[1].split('.',1)[0]
            n,g = videoname.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            self.frame_count[videoname]=dic_frame[line]

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample20()
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        self.dic_training_spatial={}
        self.dic_training_motion = {}
        for video in self.train_video:
            #print videoname
            nb_frame = self.frame_count[video]-10+1
            nb_clips = self.frame_count[video]-10+1
            key_spatial = video+' '+ str(nb_frame)
            key_motion = video +' ' + str(nb_clips)
            self.dic_training_spatial[key_spatial] = self.train_video[video]
            self.dic_training_motion[key_motion] = self.train_video[video]


    def val_sample20(self):
        print '==> sampling testing frames'
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video]-10+1
            interval = int(nb_frame/19)
            for i in range(19):
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]

    def val_sample19(self):
        self.dic_test_idx = {}
        #print len(self.test_video)
        for video in self.test_video:
            n,g = video.split('_',1)

            sampling_interval = int((self.frame_count[video]-10+1)/19)
            for index in range(19):
                clip_idx = index*sampling_interval
                key = video + ' ' + str(clip_idx+1)
                self.dic_test_idx[key] = self.test_video[video]


    def train(self):
        training_set = fusion_dataset(dic_spatial=self.dic_training_spatial, dic_motion=self.dic_training_motion, in_channel=self.in_channel, root_dir_spatial=self.data_path_spatial, root_dir_motion=self.data_path_motion, mode='train', transform_spatial = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]),
            transform_motion = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ])
        )
        print '==> Training data :',len(training_set),'frames/videos'
        print training_set[1][0]['img1'].size()

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )
        return train_loader

    def validate(self):
        validation_set = fusion_dataset(dic_spatial=self.dic_training_spatial, dic_motion=self.dic_training_motion, in_channel=self.in_channel, root_dir_spatial=self.data_path_spatial, root_dir_motion=self.data_path_motion, mode='val', transform_spatial = transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]),
            transform_motion = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ])
        )

        print '==> Validation data :',len(validation_set),'frames'
        print validation_set[1][1].size()

        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader





if __name__ == '__main__':

    dataloader = fusion_dataloader(BATCH_SIZE=1, num_workers=1, in_channel = 10,
                                path_spatial='/home/ubuntu/data/UCF101/spatial_no_sampled/',
                                path_motion='/home/ubuntu/data/UCF101/spatial_no_sampled/',
                                ucf_list='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/github/UCF_list/',
                                ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()
