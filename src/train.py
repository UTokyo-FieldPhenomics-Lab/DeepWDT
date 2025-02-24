import argparse
import os
import random
import time
from datetime import datetime

import petname
import yaml
import torch

from dataset import build_dataset, CollateFunction
from evaluation import eval_model
from models import build_yowo_model
from pre_processing import Augmentation, BaseTransform
from solver import build_loss, build_optimizer, get_lr_scheduler, set_optimizer_lr
from utils import distributed_utils
from utils.dummy_mlflow import NoOpMLflow
from utils.log import print_log


def parse_args():
    parser = argparse.ArgumentParser(description='Waggle Dance Translation')

    parser.add_argument('--training_config', default='src/config/parameters.yaml',
                        type=str, help='Path to the training config yaml file.')

    return parser.parse_args()


def train(parameters, models_architecture, run_name):
    print("Arguments: ", parameters)
    print('-----------------------------------------------------------------------------------------------------------')

    # Instantiate the training dataset and dataloader
    print('Loading the training dataset...')
    training_transform = Augmentation(
        img_size   = parameters['TRAIN']['DATASET']['IMAGE_SIZE'],
        jitter     = parameters['TRAIN']['AUGMENTATION']['JITTER'],
        hue        = parameters['TRAIN']['AUGMENTATION']['HUE'],
        saturation = parameters['TRAIN']['AUGMENTATION']['SATURATION'],
        exposure   = parameters['TRAIN']['AUGMENTATION']['EXPOSURE']
    )

    training_dataset = build_dataset(
        parameters = parameters['TRAIN']['DATASET'],
        transform = training_transform,
        split = 'train'
    )

    training_dataloader = torch.utils.data.DataLoader(
        dataset     = training_dataset,
        collate_fn  = CollateFunction(),
        num_workers = parameters['TRAIN']['NUM_WORKERS'],
        batch_size  = parameters['TRAIN']['BATCH_SIZE'],
        shuffle     = True,
    )
    print('Training dataset loaded!')

    # Instantiate the validation dataset and dataloader
    print('Loading the validation dataset...')
    validation_dataset = build_dataset(
        parameters = parameters['TRAIN']['DATASET'],
        transform = BaseTransform(img_size=parameters['TRAIN']['DATASET']['IMAGE_SIZE']),
        split = 'val'
    )

    evaluation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=parameters['EVAL']['BATCH_SIZE'],
        collate_fn=CollateFunction(),
        num_workers=parameters['EVAL']['NUM_WORKERS'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    print('Validation dataset loaded!')

    # Instantiate the model
    print('Building the model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_yowo_model(
        parameters         = parameters['MODEL'],
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
                           num_class  = len(parameters['TRAIN']['DATASET']['CLASS']),
                           center_sampling_radius = parameters['MODEL']['CENTER_SAMPLING_RADIUS'],
                           topk = parameters['MODEL']['TOP_K'])

    optimizer, start_epoch = build_optimizer(
        parameters['TRAIN']['SOLVER'],
        model,
        0,)
    print('Optimizer built!')

    print('Training...')
    # Training loop
    max_epoch = parameters['TRAIN']['MAX_EPOCH']
    epoch_size = len(training_dataloader)
    for epoch in range(start_epoch, max_epoch):
        t0 = time.time()

        for iter_i, (frame_ids, video_clips, targets) in enumerate(training_dataloader):
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
            losses /= parameters['TRAIN']['SOLVER']['ACCUMULATE']
            losses.backward()
            if ni % parameters['TRAIN']['SOLVER']['ACCUMULATE'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Logs
            if iter_i % 10 == 0:
                t1 = time.time()
                delta = t1-t0

                # Console log
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                print_log(cur_lr,
                          epoch,
                          max_epoch,
                          iter_i,
                          epoch_size,
                          loss_dict_reduced,
                          delta,
                          parameters['TRAIN']['SOLVER']['ACCUMULATE'])

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
            lr_scheduler_func = get_lr_scheduler(
                lr=parameters['TRAIN']['SOLVER']['BASE_LR'],
                warmup_total_iters = parameters['TRAIN']['SOLVER']['WARMUP_EPOCH']*epoch_size,
                no_aug_iters = parameters['TRAIN']['SOLVER']['NO_DECREASE_LR_EPOCH']*epoch_size,
                total_iters = max_epoch*epoch_size,
                warmup_lr_start = 0,
                min_lr_ratio = parameters['TRAIN']['SOLVER']['MIN_LR_RATIO'])
            set_optimizer_lr(optimizer, lr_scheduler_func, ni)

        # Save the model
        version = parameters['MODEL']['VERSION'].split('_')[-1]
        len_clip = parameters['TRAIN']['DATASET']
        path_to_save = os.path.join(f'runs/train/{run_name}/weights', f'{version}_K{len_clip}')
        os.makedirs(path_to_save, exist_ok=True)

        weight_name = f'epoch_{epoch+1}.pth'
        checkpoint_path = os.path.join(path_to_save, weight_name)
        torch.save(
            {
                'model': model.state_dict(),
                'epoch': epoch,
                'args': parameters
            },
            checkpoint_path)

        # Evaluate the model
        # print('Evaluating the model...')
        # model.eval()
        # model.trainable = False
        # eval_model(evaluation_dataloader, validation_dataset.tubes, model, parameters['EVAL'], device)
        # model.train()
        # model.trainable = True


if __name__ == '__main__':
    args = parse_args()

    # Load configs
    path_training_config = args.training_config
    with open(path_training_config) as f:
        parameters = yaml.safe_load(f)
    path_model_config = "src/config/models.yaml"
    with open(path_model_config) as f:
        models_architecture = yaml.safe_load(f)

    # Define experiment name
    random_name = petname.Generate(words=2, separator="-")
    random_number = random.randint(1000, 9999)
    run_name = f"{datetime.now().strftime('%y%m%d-%H%M%S')}-{random_name}-{random_number}"
    run_path = f'runs/{run_name}'
    os.makedirs(run_path)

    # Set MLflow environment
    if parameters['TRAIN']['MLFLOW']:
        import mlflow.pytorch
        mlflow.set_experiment('Deep Waggle Dance Translation')
    else:
        mlflow = NoOpMLflow(run_name)

    # Main loop
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact(path_training_config)
        mlflow.log_artifact(path_model_config)

        train(parameters, models_architecture, run_name)
