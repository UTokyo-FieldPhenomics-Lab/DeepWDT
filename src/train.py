import cv2
import os
import random
import time
from datetime import datetime

import petname
import yaml
import torch
# from evaluator.eval_recognition import eval as eval
# from evaluator.eval_tracking import eval_tracking

from dataset import build_2d_dataset, CollateFunction
from models import build_model
from preprocessing import Augmentation
from solver import build_loss, build_optimizer, get_lr_scheduler, set_optimizer_lr
from utils import distributed_utils
from utils.dummy_mlflow import NoOpMLflow
from utils.log import print_log


def train(parameters, models_architecture, run_name):
    global_path = (os.getcwd())

    print("Arguments: ", parameters)
    print('-----------------------------------------------------------------------------------------------------------')

    # Instantiate the training dataset and dataloader
    print('Loading the dataset...')
    training_transform = Augmentation(
        img_size   = parameters['TRAIN']['DATASET']['IMAGE_SIZE'],
        jitter     = parameters['TRAIN']['AUGMENTATION']['JITTER'],
        hue        = parameters['TRAIN']['AUGMENTATION']['HUE'],
        saturation = parameters['TRAIN']['AUGMENTATION']['SATURATION'],
        exposure   = parameters['TRAIN']['AUGMENTATION']['EXPOSURE']
    )

    training_dataset = build_2d_dataset(parameters = parameters['TRAIN']['DATASET'],
                                     transform = training_transform,
                                     split = 'train')

    dataloader = torch.utils.data.DataLoader(
        dataset     = training_dataset,
        collate_fn  = CollateFunction(),
        num_workers = parameters['TRAIN']['NUM_WORKERS'],
        batch_size  = parameters['TRAIN']['BATCH_SIZE'],
        shuffle     = True,
    )
    print('Dataset loaded!')

    # Instantiate the model
    print('Building the model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    model = build_model(parameters         = parameters['MODEL'],
                        model_architecture = models_architecture[parameters['MODEL']['VERSION']],
                        nb_class           = len(parameters['TRAIN']['DATASET']['CLASS'])+1,
                        device             = device,
                        trainable          = True)
    model = model.to(device).train()
    print('Model built!')

    # Optimizer & warmup scheduler
    print('Building the optimizer...')
    criterion = build_loss(parameters = parameters['TRAIN']['SOLVER'],
                           img_size   = parameters['TRAIN']['DATASET']['IMAGE_SIZE'],
                           num_class  = len(parameters['TRAIN']['DATASET']['CLASS'])+1,
                           center_sampling_radius = parameters['MODEL']['CENTER_SAMPLING_RADIUS'],
                           topk = parameters['MODEL']['TOP_K'])
    base_lr = parameters['TRAIN']['SOLVER']['BASE_LR']
    min_lr_ratio = parameters['TRAIN']['SOLVER']['MIN_LR_RATIO']
    warmup_epoch = parameters['TRAIN']['SOLVER']['WARMUP_EPOCH']
    no_decrease_lr_epoch = parameters['TRAIN']['SOLVER']['NO_DECREASE_LR_EPOCH']
    accumulate = parameters['TRAIN']['SOLVER']['ACCUMULATE']
    optimizer, start_epoch = build_optimizer(parameters['TRAIN']['SOLVER'], model, 0,)
    print('Optimizer built!')

    print('Training...')
    # Training loop
    max_epoch = parameters['TRAIN']['MAX_EPOCH']
    epoch_size = len(dataloader)
    for epoch in range(start_epoch, max_epoch):
        t0 = time.time()

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
                print('The loss is NAN, continue.')
                continue

            # Optimize
            losses /= accumulate
            losses.backward()
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Log
            if iter_i % 10 == 0:
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
        version = parameters['MODEL']['MODEL']['VERSION'].split('_')[-1]
        len_clip = parameters['DATASET']['LEN_CLIP']
        path_to_save = os.path.join(f'runs/train/{run_name}/weights', f'{version}_K{len_clip}')

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        weight_name = f'epoch_{epoch+1}.pth'
        checkpoint_path = os.path.join(path_to_save, weight_name)
        torch.save({'model': model.state_dict(),
                    'epoch': epoch,
                    'args': parameters},
                   checkpoint_path)

        # Evaluate the model
        model.eval()
        model.trainable = False

        # if parameters['RECOGNITION']['FMAP']:
        #     fmap = eval(
        #         parameters=parameters,
        #         model=model,
        #         transform=BaseTransform(img_size=parameters['IMAGE_SIZE']),
        #         collate_fn=CollateFunc(),
        #         metric='fmap',
        #         run_name=run_name,
        #     )
        #     mlflow.log_metric('frame_map_0.5', float(fmap[0]), step=epoch)
        # os.chdir(global_path)
        #
        # if parameters['RECOGNITION']['VMAP']:
        #     vmap = eval(
        #         parameters=parameters,
        #         model=model,
        #         transform=BaseTransform(img_size=parameters['IMAGE_SIZE']),
        #         collate_fn=CollateFunc(),
        #         metric='vmap',
        #         run_name=run_name,
        #     )
        #     mlflow.log_metric('video_map_0.3', float(vmap), step=epoch)

        # if parameters['TRACKING']:
        #
        #     checkpoint_path = 'epoch_20.pth'
        #     state_dict = torch.load(checkpoint_path)
        #     model.load_state_dict(state_dict)
        #
        #     eval_tracking(parameters, model, device, run_name)

        model.train()
        model.trainable = True


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

    if parameters['TRAIN']['MLFLOW']:
        import mlflow.pytorch
        mlflow.set_experiment('Deep Waggle Dance Translation')
    else:
        mlflow = NoOpMLflow(run_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact("src/config/parameters.yaml")
        mlflow.log_artifact("src/config/models.yaml")

        train(parameters, models_architecture, run_name)
