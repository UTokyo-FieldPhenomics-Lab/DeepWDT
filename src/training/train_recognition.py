import argparse
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from copy import deepcopy
import random
import os
import time
from datetime import datetime

import petname
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.dummy_mlflow import NoOpMLflow
from src.utils import distributed_utils
from src.utils.com_flops_params import FLOPs_and_Params
from src.utils.misc import CollateFunc, build_dataset, build_dataloader
from src.utils.solver.optimizer import build_optimizer
from src.utils.solver.warmup_schedule import get_lr_scheduler, set_optimizer_lr
from src.config import build_dataset_config, build_model_config
from src.models import build_model


GLOBAL_SEED = 42


def train(parameters, models_architecture):
    print("Arguments: ", parameters)

    # device used for training
    device = torch.device(parameters['DEVICE'])

    # dataset and evaluator
    dataset, _, num_classes = build_dataset(parameters, is_train=True)

    # dataloader
    dataloader = build_dataloader(parameters, dataset, CollateFunc(), is_train=True)

    # build model
    model, criterion = build_model(parameters, models_architecture[parameters['MODEL_VERSION']], trainable=True)

    model = model.to(device).train()

    # DDP
    model_without_ddp = model
    if parameters['DISTRIBUTION']['DISTRIBUTED']:
        model = DDP(model, device_ids=[parameters['GPU']])
        model_without_ddp = model.module

    # SyncBatchNorm
    if parameters['DISTRIBUTION']['SYBN'] and parameters['DISTRIBUTION']['DISTRIBUTED']:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # optimizer & warmup scheduler
    base_lr = parameters['SOLVER']['BASE_LR']
    min_lr_ratio = parameters['SOLVER']['MIN_LR_RATIO']
    warmup_epoch = parameters['SOLVER']['WARMUP_EPOCH']
    no_decrease_lr_epoch = parameters['SOLVER']['NO_DECREASE_LR_EPOCH']
    accumulate = parameters['SOLVER']['ACCUMULATE']
    optimizer, start_epoch = build_optimizer(parameters['SOLVER'], model_without_ddp, 0, parameters['RESUME'])

    # training configuration
    max_epoch = parameters['MAX_EPOCH']
    epoch_size = len(dataloader)

    # start to train
    t0 = time.time()
    for epoch in range(start_epoch, max_epoch):
        if parameters['DISTRIBUTION']['DISTRIBUTED']:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (frame_ids, video_clips, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size

            # to device
            video_clips = video_clips.to(device)

            # inference
            outputs = model(video_clips)
            
            # loss
            loss_dict = criterion(outputs, targets)
            losses = loss_dict['losses']

            # reduce            
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue

            # Backward
            losses /= accumulate
            losses.backward()

            # Optimize
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                    
            # Display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                print_log(cur_lr, epoch,  max_epoch, iter_i, epoch_size,loss_dict_reduced, t1-t0, accumulate)
            
                t0 = time.time()

            # warmup
            lr_scheduler_func = get_lr_scheduler(lr=base_lr, warmup_total_iters = warmup_epoch*epoch_size, no_aug_iters = no_decrease_lr_epoch*epoch_size, total_iters = max_epoch*epoch_size, warmup_lr_start = 0, min_lr_ratio = min_lr_ratio)
            set_optimizer_lr(optimizer, lr_scheduler_func, ni)


        # save model
        version = parameters['MODEL_VERSION'].split('_')[-1]
        len_clip = parameters['LEN_CLIP']
        path_to_save = os.path.join('runs/training/weights', f'{version}_K{len_clip}')

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        weight_name = f'epoch_{epoch+1}.pth'
        checkpoint_path = os.path.join(path_to_save, weight_name)
        torch.save({'model': model_without_ddp.state_dict(),
                            'epoch': epoch,
                            'args': parameters},
                            checkpoint_path)  

def print_log(lr, epoch, max_epoch, iter_i, epoch_size, loss_dict, time, accumulate):
    # basic infor
    log =  '[Epoch: {}/{}]'.format(epoch+1, max_epoch)
    log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
    log += '[lr: {:.6f}]'.format(lr[0])
    # loss infor
    for k in loss_dict.keys():
        if k == 'losses':
            log += '[{}: {:.2f}]'.format(k, loss_dict[k] * accumulate)
        else:
            log += '[{}: {:.2f}]'.format(k, loss_dict[k])

    # other infor
    log += '[time: {:.2f}]'.format(time)

    # print log infor
    print(log, flush=True)


if __name__ == '__main__':
    with open("src/config/parameters.yaml") as f:
        parameters = yaml.safe_load(f)

    with open("src/config/models.yaml") as f:
        models_architecture = yaml.safe_load(f)

    if parameters['MLFLOW']:
        import mlflow.pytorch
        mlflow.set_experiment('Deep Waggle Dance Translation')
    else:
        mlflow = NoOpMLflow()

    random_name = petname.Generate(words=2, separator="-")
    random_number = random.randint(1000, 9999)
    run_name = f"{datetime.now().strftime('%y%m%d-%H%M%S')}-{random_name}-{random_number}"
    run_path = f'runs/train/{run_name}'
    os.makedirs(run_path)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact("src/config/parameters.yaml")
        mlflow.log_artifact("src/config/models.yaml")

        train(parameters, models_architecture)
