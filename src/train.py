import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.dataset import build_dataset, CollateFunction, TrainTransform, EvalTransform
from src.evaluation import get_metrics
from src.model import build_yowo_model, track
from src.solver import build_loss, build_optimizer, get_lr_scheduler, set_optimizer_lr
from src.utils import distributed_utils, thieve_confidence, grouped_nms, NoOpMLflow, print_log
from src.config import load_configuration


def train_function(run_name, path_configuration):
    configuration = load_configuration(path_configuration)
    train_configuration = configuration.train
    eval_configuration = configuration.eval

    print("Arguments: ", configuration)
    print('-----------------------------------------------------------------------------------------------------------')

    # Instantiate the training dataset and dataloader
    print('Loading the training dataset...')
    training_transform = TrainTransform(
        img_size=train_configuration.dataset.image_size[0],
        jitter=train_configuration.augmentation.jitter,
        hue=train_configuration.augmentation.hue,
        saturation=train_configuration.augmentation.saturation,
        exposure=train_configuration.augmentation.exposure
    )

    training_dataset = build_dataset(
        configuration=train_configuration.dataset,
        transform=training_transform,
        split='train'
    )

    training_dataloader = torch.utils.data.DataLoader(
        dataset=training_dataset,
        collate_fn=CollateFunction(),
        num_workers=train_configuration.num_workers,
        batch_size=train_configuration.batch_size,
        shuffle=True,
    )
    print('Training dataset loaded!')

    # Instantiate the validation dataset and dataloader
    print('Loading the validation dataset...')
    validation_dataset = build_dataset(
        configuration=eval_configuration.dataset,
        transform=EvalTransform(img_size=train_configuration.dataset.image_size[0]),
        split='val'
    )

    evaluation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=eval_configuration.batch_size,
        collate_fn=CollateFunction(),
        num_workers=eval_configuration.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    print('Validation dataset loaded!')

    # Instantiate the model
    print('Building the model...')
    device = train_configuration.device
    model = build_yowo_model(model_configuration=train_configuration.model,
                             centered_clip=train_configuration.dataset.centered_clip,
                             device=device,
                             trainable=True)
    model = model.to(device).train()
    print('Model built!')

    # Optimizer & warmup scheduler
    print('Building the optimizer...')
    criterion = build_loss(configuration = train_configuration)

    optimizer, start_epoch = build_optimizer(
        train_configuration.solver,
        model,
        0,)
    print('Optimizer built!')

    # Set MLflow environment
    if train_configuration.mlflow:
        import mlflow.pytorch
        mlflow.set_experiment('Deep Waggle Dance Translation')
    else:
        mlflow = NoOpMLflow(run_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_artifact(path_configuration)

        # Training loop
        print('Training...')
        max_epoch = train_configuration.max_epoch
        epoch_size = len(training_dataloader)
        for epoch in range(start_epoch, max_epoch):

            for iter_i, (frame_ids, video_clips, targets) in enumerate(training_dataloader):
                ni = iter_i + epoch * epoch_size

                # Model inference
                video_clips = video_clips.to(device)
                t0 = time.time()
                outputs = model(video_clips)

                # Loss calculation
                loss_dict = criterion(outputs, targets)
                losses = loss_dict['losses']
                loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)
                if torch.isnan(losses):
                    print('The loss is NAN, continue.')
                    continue

                # Optimize
                losses /= train_configuration.solver.accumulate
                losses.backward()
                if ni % train_configuration.solver.accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                t1 = time.time()

                # Logs
                if iter_i % 10 == 0:
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
                              train_configuration.solver.accumulate)

                    # MLflow log
                    mlflow.log_metric('lr', cur_lr[0], step=ni)
                    for k in loss_dict_reduced.keys():
                        if k == 'losses':
                            mlflow.log_metric(k, loss_dict_reduced[k] * train_configuration.solver.accumulate, step=ni)
                        else:
                            mlflow.log_metric(k, loss_dict_reduced[k], step=ni)
                    mlflow.log_metric('time_in_seconds', delta, step=ni)
                    t0 = time.time()

                # LR scheduler update
                lr_scheduler_func = get_lr_scheduler(
                    lr=train_configuration.solver.base_lr,
                    warmup_total_iters = train_configuration.solver.warmup_epoch*epoch_size,
                    no_aug_iters = train_configuration.solver.no_decrease_lr_epoch*epoch_size,
                    total_iters = max_epoch*epoch_size,
                    warmup_lr_start = 0,
                    min_lr_ratio = train_configuration.solver.min_lr_ratio)
                set_optimizer_lr(optimizer, lr_scheduler_func, ni)

            # Save the model
            path_to_save = os.path.join(f'runs/train/{run_name}/weights')
            os.makedirs(path_to_save, exist_ok=True)
            weight_name = f'epoch_{epoch+1}.pth'
            checkpoint_path = os.path.join(path_to_save, weight_name)
            torch.save(
                {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'args': configuration
                },
                checkpoint_path)

            print('Computational time assessment...')
            print(f'[2D backbone mean inference time: {round(np.mean(model.benchmark_backbone2D[1:])*1000,2)}]'
                  f'[2D backbone stdv inference time: {round(np.std(model.benchmark_backbone2D[1:])*1000,2)}]')
            print(f'[3D backbone mean inference time: {round(np.mean(model.benchmark_backbone3D[1:])*1000,2)}]'
                  f'[3D backbone stdv inference time: {round(np.std(model.benchmark_backbone3D[1:])*1000,2)}]')
            print(f'[Head mean inference time: {round(np.mean(model.benchmark_head[1:])*1000,2)}]'
                  f'[Head stdv inference time: {round(np.std(model.benchmark_head[1:])*1000,2)}]')

            # Evaluate the model
            print('Evaluating the model...')
            model.eval()
            model.trainable = False

            # Detect dancing bees
            print('1. Detecting dancing bees...')

            results_list = []
            for iter_i, (frame_ids, video_clips, targets) in enumerate(tqdm(evaluation_dataloader)):

                # Model inference
                video_clips = video_clips.to(device)
                outputs = model(video_clips)  # scores, labels, bboxes

                # Add results to a dataframe
                scores, labels, bboxes = outputs
                for index, (score, label, bboxe) in enumerate(zip(scores, labels, bboxes)):

                    for detection_id in range(len(score)):
                        results_list.append({
                            'video': targets[index]['video'],
                            'frame_id': int(targets[index]['image_id']),
                            'class': int(label[detection_id]),
                            'x0': bboxe[detection_id][0],
                            'y0': bboxe[detection_id][1],
                            'x1': bboxe[detection_id][2],
                            'y1': bboxe[detection_id][3],
                            'confidence': score[detection_id]
                        })

            detections = pd.DataFrame(results_list)
            detections['frame_id'] = detections['frame_id'].astype(int)
            detections['class'] = detections['class'].astype(int)

            # Thieve results based on the confidence
            detections = thieve_confidence(detections, eval_configuration.confidence_threshold)

            # Apply NMS to results
            detections = grouped_nms(detections, eval_configuration.nms_threshold)

            # Track dancing bees through frames
            print('2. Tracking dancing bees...')
            detections = track(detections,
                               iou_threshold = eval_configuration.track_iou_threshold,
                               duration_threshold = eval_configuration.track_duration_threshold)

            # Compute evaluation metrics
            print('3. Computing evaluation metrics...')
            if len(detections) == 0:
                eval_metrics = {'angle_rmse': -1, 'duration_rmse': -1, 'detected_runs': 0}
            else:
                detections[['x0', 'x1', 'y0', 'y1']] = detections[['x0', 'x1', 'y0', 'y1']] * 224
                eval_metrics = get_metrics(detections, validation_dataset.df)
                print(f'[Detected runs: {eval_metrics["detected_runs"]}%][Angle RMSE: {eval_metrics["angle_rmse"]}][Duration RMSE: {eval_metrics["duration_rmse"]}]')

            # Log evaluation metrics
            mlflow.log_metric('angle_rmse', eval_metrics['angle_rmse'], step=epoch)
            mlflow.log_metric('duration_rmse', eval_metrics['duration_rmse'], step=epoch)
            mlflow.log_metric('detected_runs', eval_metrics['detected_runs'], step=epoch)
            model.train()
            model.trainable = True
