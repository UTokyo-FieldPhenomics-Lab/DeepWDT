import os

import torch
import torch.nn as nn

# from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
# from dataset.ava import AVA_Dataset
from src.dataset.training_dataset import Training_Dataset
from src.dataset.transforms import Augmentation, BaseTransform

from src.evaluator.training_dataset_evaluator import TRAINING_DATASET_Evaluator
# from src.evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
# from src.evaluator.ava_evaluator import AVA_Evaluator


def build_dataset(parameters, is_train=False, eval_split=None):
    # transform
    augmentation = Augmentation(
        img_size=parameters['IMAGE_SIZE'],
        jitter=parameters['AUGMENTATION']['JITTER'],
        hue=parameters['AUGMENTATION']['HUE'],
        saturation=parameters['AUGMENTATION']['SATURATION'],
        exposure=parameters['AUGMENTATION']['EXPOSURE']
        )
    basetransform = BaseTransform(
        img_size=parameters['IMAGE_SIZE'],
        )

    # dataset
    if parameters['DATASET'] == 'training_dataset':
        data_dir = os.path.join('data/', 'training_dataset')
        
        # dataset
        dataset = Training_Dataset(
            data_root=data_dir,
            dataset=parameters['DATASET'],
            img_size=parameters['IMAGE_SIZE'],
            transform=augmentation,
            is_train=is_train,
            len_clip=parameters['LEN_CLIP'],
            sampling_rate=parameters['SAMPLING_RATE'],
            eval_split=eval_split
            )
        
        num_classes = dataset.num_classes

        # evaluator
        evaluator = TRAINING_DATASET_Evaluator(
            data_root=data_dir,
            dataset=parameters['DATASET'],
            model_name=parameters['MODEL_VERSION'],
            metric='fmap',
            img_size=parameters['IMAGE_SIZE'],
            len_clip=parameters['LEN_CLIP'],
            batch_size=parameters['BATCH_SIZE']['EVAL'],
            conf_thresh=parameters['CONF_THRESH'],
            iou_thresh=parameters['IOU_THRESH'],
            gt_folder='src/evaluator/training_dataset_evaluation/',
            save_path='./evaluator/eval_results/',
            transform=basetransform,
            collate_fn=CollateFunc(),
            eval_split=eval_split          
        )

    else:
        print('unknow dataset.')
        exit(0)

    print('==============================')
    print('Training model on:', parameters['DATASET'])
    print('The dataset size:', len(dataset))

    return dataset, evaluator, num_classes


def build_dataloader(parameters, dataset, collate_fn=None, is_train=False):

    if is_train:
        sampler = torch.utils.data.RandomSampler(dataset)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                            parameters['BATCH_SIZE']['TRAIN'],
                                                            drop_last=True)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_sampler=batch_sampler_train,
            collate_fn=collate_fn, 
            num_workers=parameters['NUM_WORKERS'],
            pin_memory=True
            )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=parameters['NUM_WORKERS'],
            drop_last=False,
            pin_memory=True
            )
    
    return dataloader
    

def load_weight(model, path_to_ckpt=None):
    if path_to_ckpt is None:
        print('No trained weight ..')
        return model
        
    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class CollateFunc(object):
    def __call__(self, batch):
        batch_frame_id = []
        batch_key_target = []
        batch_video_clips = []

        for sample in batch:
            key_frame_id = sample[0]
            video_clip = sample[1]
            key_target = sample[2]
            
            batch_frame_id.append(key_frame_id)
            batch_video_clips.append(video_clip)
            batch_key_target.append(key_target)

        # List [B, 3, T, H, W] -> [B, 3, T, H, W]
        batch_video_clips = torch.stack(batch_video_clips)
        
        return batch_frame_id, batch_video_clips, batch_key_target
