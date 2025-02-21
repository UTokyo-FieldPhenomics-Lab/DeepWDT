import os
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from utils.box_ops import rescale_bboxes
from .cal_frame_mAP import evaluate_frameAP
from .cal_video_mAP import evaluate_videoAP


def evaluate_frame_map(model, dataloader, run_name, epoch, model_name, len_clip, split, iou_thresh, show_pr_curve=False):
    print('Evaluating Frame mAP ...')

    path_model_outputs = f'runs/train/{run_name}/model_outputs'
    save_path = f'runs/eval/{run_name}'
    gt_folder = os.path.join('data/training_dataset/evaluation', split)

    if not os.path.exists(path_model_outputs):
        os.makedirs(path_model_outputs)
    else:
        shutil.rmtree(path_model_outputs)
        os.makedirs(path_model_outputs)

    # inference
    for iter_i, (batch_frame_id, batch_video_clip, batch_target) in enumerate(tqdm(dataloader)):
        # to device
        batch_video_clip = batch_video_clip.to(model.device)

        with torch.no_grad():
            # inference
            batch_scores, batch_labels, batch_bboxes = model(batch_video_clip)

            # process batch
            for bi in range(len(batch_scores)):
                frame_id = batch_frame_id[bi]
                scores = batch_scores[bi]
                labels = batch_labels[bi]
                bboxes = batch_bboxes[bi]
                target = batch_target[bi]

                # rescale bbox
                orig_size = target['orig_size']
                bboxes = rescale_bboxes(bboxes, orig_size)

                detection_path = os.path.join(path_model_outputs, frame_id)

                with open(detection_path, 'w+') as f_detect:
                    for score, label, bbox in zip(scores, labels, bboxes):
                        x1 = round(bbox[0])
                        y1 = round(bbox[1])
                        x2 = round(bbox[2])
                        y2 = round(bbox[3])
                        cls_id = int(label) + 1

                        f_detect.write(
                            str(cls_id) + ' ' + str(score) + ' ' \
                            + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')

    working_directory = os.getcwd()

    metric_list = evaluate_frameAP(
        gt_folder,
        path_model_outputs,
        iou_thresh,
        save_path,
        'training_dataset',
        show_pr_curve)

    os.chdir(working_directory)

    csv_save_path = os.path.join(os.path.dirname(os.getcwd()), "evalf.csv")

    if not os.path.exists(os.path.dirname(csv_save_path)):
        os.makedirs(os.path.dirname(csv_save_path))

    if os.path.exists(csv_save_path):
        df_valmap = pd.read_csv(csv_save_path)
    else:
        df_valmap = pd.DataFrame()

    dict_results = {'version': [model_name],
                    'K': [len_clip],
                    'epoch': [epoch],
                    'split': [split],
                    'mAP': [metric_list[1][5:-1]],
                    }

    df_valmap = pd.concat([df_valmap, pd.DataFrame(dict_results)], ignore_index=True)
    df_valmap.to_csv(csv_save_path, index=False)

    return [metric_list[1][5:-1]]


def evaluate_video_map(model, dataloader, run_name, epoch, model_name, len_clip, split,):
    print('Evaluating Video mAP ...')

    video_testlist = []

    with open(os.path.join('data/training_dataset/evaluation', split), 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.rstrip()
            video_testlist.append(line)

    detected_boxes = {}
    gt_videos = {}

    # get ground truth
    gt_data = loadmat('data/training_dataset/splitfiles/finalAnnots.mat')['annot']
    n_videos = gt_data.shape[1]

    for i in range(n_videos):
        video_name = str(gt_data[0][i][1][0]).strip()

        if video_name in video_testlist:
            n_tubes = len(gt_data[0][i][2][0])
            v_annotation = {}
            all_gt_boxes = []

            for j in range(n_tubes):
                gt_one_tube = []
                tube_start_frame = gt_data[0][i][2][0][j][1][0][0]
                tube_end_frame = gt_data[0][i][2][0][j][0][0][0]
                tube_class = gt_data[0][i][2][0][j][2][0][0]
                tube_data = gt_data[0][i][2][0][j][3]
                tube_length = tube_end_frame - tube_start_frame

                for k in range(tube_length):
                    gt_boxes = []
                    gt_boxes.append(int(tube_start_frame + k))
                    gt_boxes.append(float(tube_data[k][0]))
                    gt_boxes.append(float(tube_data[k][1]))
                    gt_boxes.append(float(tube_data[k][2]))
                    gt_boxes.append(float(tube_data[k][3]))
                    gt_one_tube.append(gt_boxes)
                all_gt_boxes.append(gt_one_tube)

            v_annotation['gt_classes'] = tube_class
            v_annotation['tubes'] = np.array(all_gt_boxes)
            gt_videos[video_name] = v_annotation

    # inference
    for i, line in enumerate(tqdm(lines)):
        line = line.rstrip()

        # set video
        self.testset.set_video_data(line)

        # dataloader
        self.testloader = torch.utils.data.DataLoader(
            dataset=self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )

        for iter_i, (batch_img_name, batch_video_clip, batch_target) in enumerate(self.testloader):
            # to device
            batch_video_clip = batch_video_clip.to(model.device)

            with torch.no_grad():
                # inference
                batch_scores, batch_labels, batch_bboxes = model(batch_video_clip)

                # process batch
                for bi in range(len(batch_scores)):
                    img_name = batch_img_name[bi]
                    scores = batch_scores[bi]
                    labels = batch_labels[bi]
                    bboxes = batch_bboxes[bi]
                    target = batch_target[bi]

                    # rescale bbox
                    orig_size = target['orig_size']
                    bboxes = rescale_bboxes(bboxes, orig_size)

                    img_annotation = {}
                    for cls_idx in range(self.num_classes):
                        inds = np.where(labels == cls_idx)[0]
                        c_bboxes = bboxes[inds]
                        c_scores = scores[inds]
                        # [n_box, 5]
                        boxes = np.concatenate([c_bboxes, c_scores[..., None]], axis=-1)
                        img_annotation[cls_idx + 1] = boxes
                    detected_boxes[img_name] = img_annotation

    iou_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]

    csv_save_path = os.path.join(f"runs/train/{run_name}", "evalv.csv")

    if not os.path.exists(os.path.dirname(csv_save_path)):
        os.makedirs(os.path.dirname(csv_save_path))

    if os.path.exists(csv_save_path):
        df_valmap = pd.read_csv(csv_save_path)
    else:
        df_valmap = pd.DataFrame()

    to_add = {'version': self.model_name,
              'K': self.len_clip,
              'epoch': self.epoch,
              'split': self.eval_split,
              # 'time':datetime.now().strftime("%d/%m/%Y %H:%M")
              }

    for iou_th in iou_list:
        per_ap = evaluate_videoAP(gt_videos, detected_boxes, self.num_classes, iou_th, True)
        video_mAP = sum(per_ap) / len(per_ap)
        to_add[f'{iou_th}'] = video_mAP
        # print('-------------------------------')
        # print('V-mAP @ {} IoU:'.format(iou_th))
        # print('--Per AP: ', per_ap)
        # print('--mAP: ', round(video_mAP, 2))

    df_valmap = pd.concat([df_valmap, pd.DataFrame([to_add])], ignore_index=True)
    df_valmap.to_csv(csv_save_path, index=False)

    return to_add['0.3']

