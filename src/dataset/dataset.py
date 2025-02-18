import os
import random
import numpy as np
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image


# Training Dataset
class DATASET2D(Dataset):
    def __init__(self, dataset_name, img_size, len_clip, transform, split, centered_clip):
        self.path = dataset_name
        self.data_root = os.path.join('data', self.path)
        self.num_classes = 2

        self.transform = transform

        self.split = split
        if self.split == 'train':
            self.split_list = 'trainlist.txt'
        elif self.split == 'val':
            self.split_list = 'vallist.txt'
        elif self.split == 'test':
            self.split_list = 'testlist.txt'
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.centered_clip = centered_clip

        # Load data
        with open(os.path.join(self.data_root, self.split_list), 'r') as file:
            self.file_names = file.readlines()
        self.num_samples  = len(self.file_names)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.centered_clip:
            frame_idx, video_clip, target = self.pull_item_centered(index)
        else:
            frame_idx, video_clip, target = self.pull_item(index)

        return frame_idx, video_clip, target

    def pull_item(self, index):
        """ load a data """
        assert index <= len(self), 'index range error'
        image_path = self.file_names[index].rstrip()

        img_split = image_path.split('/')  # ex. ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']

        img_id = int(img_split[-1][:5])

        label_path = os.path.join(self.data_root, img_split[0], img_split[1], img_split[2], '{:05d}.txt'.format(img_id))

        img_folder = os.path.join(self.data_root, 'rgb-images', img_split[1], img_split[2])

        max_num = len(os.listdir(img_folder))

        # load images
        video_clip = []
        for i in reversed(range(self.len_clip)):
            # make it as a loop
            img_id_temp = img_id - i
            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num

            # load a frame
            path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[1], img_split[2] ,'{:05d}.jpg'.format(img_id_temp))
            frame = Image.open(path_tmp).convert('RGB')
            ow, oh = frame.width, frame.height

            video_clip.append(frame)

            frame_id = img_split[1] + '_' +img_split[2] + '_' + img_split[3]

        # load an annotation
        if os.path.getsize(label_path):
            target = np.loadtxt(label_path)
        else:
            target = None

        # [label, x1, y1, x2, y2] -> [x1, y1, x2, y2, label]
        label = target[..., :1]
        boxes = target[..., 1:]
        target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)
            
        # transform
        video_clip, target = self.transform(video_clip, target)

        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)

        # reformat target
        target = {
            'boxes': target[:, :4].float(),      # [N, 4]
            'labels': target[:, -1].long() - 1,    # [N,]
            'orig_size': [ow, oh],
            'video_idx':frame_id[:-10]
        }

        return frame_id, video_clip, target

    def pull_item_centered(self, index):
        """ load a data """
        assert index <= len(self), 'index range error'

        image_path = self.file_names[index].rstrip()
        img_split = image_path.split('/')  # ex. ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']

        img_id = int(img_split[-1][:5])

        label_path = os.path.join(self.data_root,
                                  img_split[0],
                                  img_split[1],
                                  img_split[2],
                                  '{:05d}.txt'.format(img_id))

        img_folder = os.path.join(self.data_root,
                                  'rgb-images',
                                  img_split[1],
                                  img_split[2])

        max_num = len(os.listdir(img_folder))

        # load images
        video_clip = []
        half_len_clip = int(self.len_clip / 2)
        for offset in range(-half_len_clip, half_len_clip):

            img_id_temp = img_id + offset

            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num

            # load a frame
            path_tmp = os.path.join(self.data_root,
                                    'rgb-images',
                                    img_split[1],
                                    img_split[2]
                                    , '{:05d}.jpg'.format(img_id_temp)
                                    )

            frame = Image.open(path_tmp).convert('RGB')
            ow, oh = frame.width, frame.height

            video_clip.append(frame)

        frame_id = img_split[1] + '_' + img_split[2] + '_' + img_split[3]

        # Load an annotation
        if os.path.getsize(label_path):
            target = np.loadtxt(label_path)
        else:
            target = None

        # [label, x1, y1, x2, y2] -> [x1, y1, x2, y2, label]
        label = target[..., :1]
        boxes = target[..., 1:]
        target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)

        # transform
        video_clip, target = self.transform(video_clip, target)

        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)

        # reformat target
        target = {
            'boxes': target[:, :4].float(),  # [N, 4]
            'labels': target[:, -1].long() - 1,  # [N,]
            'orig_size': [ow, oh],
            'video_idx': frame_id[:-10]
        }

        return frame_id, video_clip, target

    def pull_anno(self, index):
        """ load a data """
        assert index <= len(self), 'index range error'
        image_path = self.file_names[index].rstrip()

        img_split = image_path.split('/')  # ex. ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        # image name
        img_id = int(img_split[-1][:5])

        # path to label
        label_path = os.path.join(self.data_root, img_split[0], img_split[1], img_split[2], '{:05d}.txt'.format(img_id))

        # load an annotation
        target = np.loadtxt(label_path)
        target = target.reshape(-1, 5)

        return target
        

# Training Video Dataset
class DATASET3D(Dataset):
    def __init__(self,
                 data_root,
                 dataset='training_dataset',
                 img_size=448,
                 transform=None,
                 len_clip=16,
                 sampling_rate=1,):
        self.data_root = data_root
        self.dataset = dataset
        self.transform = transform
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
            
        self.num_classes = 1


    def set_video_data(self, line):
        self.line = line

        # load a video
        self.img_folder = os.path.join(self.data_root, 'rgb-images', self.line)

        self.label_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.jpg')))

    def __len__(self):
        return len(self.label_paths)


    def __getitem__(self, index):
        return self.pull_item(index)


    def pull_item(self, index):
        image_path = self.label_paths[index]

        video_split = self.line.split('/')
        video_class = video_split[0]
        video_file = video_split[1]
        img_split = image_path.split('\\')
        # img_split = image_path.split('/')

        # image name
        img_id = int(img_split[-1][:5])
        max_num = len(os.listdir(self.img_folder))
        img_name = os.path.join(video_class, video_file, '{:05d}.jpg'.format(img_id))

        # load video clip
        video_clip = []
        for i in reversed(range(self.len_clip)):
            # make it as a loop
            img_id_temp = img_id - i
            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num

            # load a frame
            path_tmp = os.path.join(f'data/{self.dataset}/', 'rgb-images', video_class, video_file ,'{:05d}.jpg'.format(img_id_temp))
            frame = Image.open(path_tmp).convert('RGB')
            ow, oh = frame.width, frame.height

            video_clip.append(frame)

        # transform
        video_clip, _ = self.transform(video_clip, normalize=False)
        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)
        orig_size = [ow, oh]  # width, height

        target = {'orig_size': [ow, oh]}

        return img_name, video_clip, target


def build_2d_dataset(parameters, transform, split,):

    return DATASET2D(dataset_name = parameters['NAME'],
                     img_size = parameters['IMAGE_SIZE'],
                     len_clip = parameters['LEN_CLIP'],
                     transform = transform,
                     split = split,
                     centered_clip = parameters['CENTERED_CLIP'])

def build_3d_dataset(parameters):
    pass
