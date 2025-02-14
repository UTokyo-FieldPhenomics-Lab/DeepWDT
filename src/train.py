import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import random
import os
import time
from datetime import datetime

import petname
import yaml
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from evaluator.eval_recognition import eval as eval
from evaluator.eval_tracking import eval_tracking
from dataset.transforms import BaseTransform
from utils import distributed_utils
from utils.dummy_mlflow import NoOpMLflow
from utils.misc import CollateFunc, build_dataset, build_dataloader
from utils.log import print_log
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import get_lr_scheduler, set_optimizer_lr
from models import build_model


GLOBAL_SEED = 42


def train(parameters, models_architecture, run_name):
    global_path = (os.getcwd())
    print("Arguments: ", parameters)
    print('Training...')

    # Create dataloader
    dataset, _, num_classes = build_dataset(parameters, is_train=True)
    dataloader = build_dataloader(parameters, dataset, CollateFunc(), is_train=True)

    # Create model and criterion
    model, criterion = build_model(parameters, models_architecture[parameters['MODEL_VERSION']], trainable=True)
    device = torch.device(parameters['DEVICE'])
    model = model.to(device).train()

    # Set up distribution over GPUs
    model_without_ddp = model
    if parameters['DISTRIBUTION']['DISTRIBUTED']:
        model = DDP(model, device_ids=[parameters['GPU']])
        model_without_ddp = model.module
    if parameters['DISTRIBUTION']['SYBN'] and parameters['DISTRIBUTION']['DISTRIBUTED']:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Optimizer & warmup scheduler
    base_lr = parameters['SOLVER']['BASE_LR']
    min_lr_ratio = parameters['SOLVER']['MIN_LR_RATIO']
    warmup_epoch = parameters['SOLVER']['WARMUP_EPOCH']
    no_decrease_lr_epoch = parameters['SOLVER']['NO_DECREASE_LR_EPOCH']
    accumulate = parameters['SOLVER']['ACCUMULATE']
    optimizer, start_epoch = build_optimizer(parameters['SOLVER'], model_without_ddp, 0, parameters['RESUME'])

    # Training loop
    max_epoch = parameters['MAX_EPOCH']
    epoch_size = len(dataloader)
    for epoch in range(start_epoch, max_epoch):
        t0 = time.time()
        if parameters['DISTRIBUTION']['DISTRIBUTED']:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        for iter_i, (frame_ids, video_clips, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size

            # Model inference
            video_clips = video_clips.to(device)
            outputs = model(video_clips)

            # Loss calculation
            loss_dict = criterion(outputs, targets)
            losses = loss_dict['losses']
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue

            # Optimize
            losses /= accumulate
            losses.backward()
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Log
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                delta = t1-t0

                # Console log
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                print_log(cur_lr, epoch,  max_epoch, iter_i, epoch_size, loss_dict_reduced, delta, accumulate)

                # MLflow log
                mlflow.log_metric('lr', cur_lr[0], step=ni)
                for k in loss_dict_reduced.keys():
                    if k == 'losses':
                        mlflow.log_metric(k, loss_dict_reduced[k]* accumulate, step=ni)
                    else:
                        mlflow.log_metric(k, loss_dict_reduced[k], step=ni)
                mlflow.log_metric('time_in_seconds', delta, step=ni)
                t0 = time.time()

            # LR scheduler update
            lr_scheduler_func = get_lr_scheduler(lr=base_lr, warmup_total_iters = warmup_epoch*epoch_size, no_aug_iters = no_decrease_lr_epoch*epoch_size, total_iters = max_epoch*epoch_size, warmup_lr_start = 0, min_lr_ratio = min_lr_ratio)
            set_optimizer_lr(optimizer, lr_scheduler_func, ni)

        # Save the model
        version = parameters['MODEL_VERSION'].split('_')[-1]
        len_clip = parameters['LEN_CLIP']
        path_to_save = os.path.join(f'runs/train/{run_name}/weights', f'{version}_K{len_clip}')

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        weight_name = f'epoch_{epoch+1}.pth'
        checkpoint_path = os.path.join(path_to_save, weight_name)
        torch.save({'model': model_without_ddp.state_dict(),
                            'epoch': epoch,
                            'args': parameters},
                            checkpoint_path)

        # Evaluate the model
        model.eval()
        model.trainable = False
        if parameters['RECOGNITION']['FMAP']:
            fmap = eval(
                parameters=parameters,
                model=model,
                transform=BaseTransform(img_size=parameters['IMAGE_SIZE']),
                collate_fn=CollateFunc(),
                metric='fmap',
                run_name=run_name,
            )
        os.chdir(global_path)
        if parameters['RECOGNITION']['VMAP']:
            vmap = eval(
                parameters=parameters,
                model=model,
                transform=BaseTransform(img_size=parameters['IMAGE_SIZE']),
                collate_fn=CollateFunc(),
                metric='vmap',
                run_name=run_name,
            )
        if parameters['TRACKING']:
            eval_tracking(parameters, model, device, run_name)
        model.train()
        model.trainable = True
        mlflow.log_metric('frame_map_0.5', float(fmap[0]), step=epoch)
        mlflow.log_metric('video_map_0.3', float(vmap), step=epoch)



if __name__ == '__main__':
    with open("src/config/parameters.yaml") as f:
        parameters = yaml.safe_load(f)

    with open("src/config/models.yaml") as f:
        models_architecture = yaml.safe_load(f)

    random_name = petname.Generate(words=2, separator="-")
    random_number = random.randint(1000, 9999)
    run_name = f"{datetime.now().strftime('%y%m%d-%H%M%S')}-{random_name}-{random_number}"
    run_path = f'runs/{run_name}'
    os.makedirs(run_path)

    if parameters['MLFLOW']:
        import mlflow.pytorch
        mlflow.set_experiment('Deep Waggle Dance Translation')
    else:
        mlflow = NoOpMLflow(run_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact("src/config/parameters.yaml")
        mlflow.log_artifact("src/config/models.yaml")

        train(parameters, models_architecture, run_name)
