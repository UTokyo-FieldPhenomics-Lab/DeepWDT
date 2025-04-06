import os
import random
import numpy as np
import glob

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# Dataset
class WaggleDanceDataset(Dataset):
    def __init__(self, dataset_name, img_size, len_clip, transform, split, centered_clip):
        # Dataset information
        self.dataset_name = dataset_name
        self.path_data = os.path.join('data', self.dataset_name)
        self.num_classes = 1
        self.split = split
        self.img_size = img_size
        self.len_clip = len_clip
        self.centered_clip = centered_clip

        # Transform function
        self.transform = transform

        # Load image list
        self.df = pd.read_csv(os.path.join(self.path_data, f'{split}.csv'))
        self.video_list = self.df['video'].unique().tolist()
        frame_list = []
        for video in self.video_list:
            for frame in os.listdir(os.path.join(self.path_data, 'labels/Dancing', video)):
                frame_list.append(f'{video}/{frame}')
        self.frame_list = frame_list
        self.num_samples  = len(self.frame_list)

        # Load tube list
        self.tubes = []

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        frame = self.frame_list[index].split('/')
        video, image = frame[0], frame[1]

        image_id = int(image[:-4])

        label_path = os.path.join(self.path_data, 'labels', 'Dancing', video, '{:05d}.txt'.format(image_id))

        max_num = len(os.listdir(os.path.join(self.path_data, 'rgb-images', 'Dancing',  video)))

        # Load images
        if self.centered_clip:
            video_clip = self._pull_clip_centered(image_id, video, max_num)
        else:
            video_clip = self._pull_clip_past(image_id, video, max_num)

        # Load annotations
        if os.path.getsize(label_path):
            target = np.loadtxt(label_path)
        else:
            target = None

        # [label, x1, y1, x2, y2] -> [x1, y1, x2, y2, label]
        label = target[..., :1]
        boxes = target[..., 1:]
        target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)

        video_clip, target = self.transform(video_clip, target)

        video_clip = torch.stack(video_clip, dim=1) # List [T, 3, H, W] -> [3, T, H, W]

        if self.split == 'train':
            target = {
                'boxes': target[:, :4].float(),      # [N, 4]
                'labels': target[:, -1].long() - 1,    # [N,]
                'orig_size': self.img_size,
            }

            return image_id, video_clip, target

        else:
            target = {
                'boxes': target[:, :4].float(),  # [N, 4]
                'labels': target[:, -1].long() - 1,  # [N,]
                'orig_size': self.img_size,
                'video': video,
                'image_id': image_id,
            }

            return image_id, video_clip, target,

    def _pull_clip_past(self, image_id, video, max_num):
        video_clip = []

        for i in reversed(range(self.len_clip)):
            img_id_temp = image_id - i

            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num

            image_path_temp = os.path.join(self.path_data, 'rgb-images', 'Dancing', video,
                                           '{:05d}.jpg'.format(img_id_temp))

            frame = Image.open(image_path_temp).convert('RGB')

            video_clip.append(frame)

        return video_clip

    def _pull_clip_centered(self, image_id, video, max_num):
        video_clip = []

        half_len_clip = int(self.len_clip / 2)

        for offset in range(-half_len_clip, half_len_clip+1):

            img_id_temp = image_id + offset

            if img_id_temp < 1:
                img_id_temp = 1
            elif img_id_temp > max_num:
                img_id_temp = max_num

            image_path_temp = os.path.join(self.path_data, 'rgb-images', 'Dancing', video,
                                           '{:05d}.jpg'.format(img_id_temp))

            frame = Image.open(image_path_temp).convert('RGB')

            video_clip.append(frame)

        return video_clip


def build_dataset(configuration, transform, split,):

    return WaggleDanceDataset(
        dataset_name = configuration.name,
        img_size = configuration.image_size,
        len_clip = configuration.len_clip,
        transform = transform,
        split = split,
        centered_clip = configuration.centered_clip)
