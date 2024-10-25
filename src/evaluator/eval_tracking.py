import argparse
import math
import numpy as np
import os
import shutil
import warnings
# TO SOLVE LATER
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*DataFrameGroupBy.apply operated on the grouping columns.*")

import cv2
import pandas as pd
import plotly.graph_objects as go
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

from src.config import build_dataset_config, build_model_config
from src.dataset.transforms import BaseTransform
from src.models import build_model
from src.utils.misc import load_weight
from src.utils.box_ops import rescale_bboxes
from src.utils.tracking import thieve_durations, get_angles, track_bees
from src.utils.vis_tools import vis_detection


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2 Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=224, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('-vs', '--vis_thresh', default=0.3, type=float,
                        help='threshold for visualization')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')

    # class label config
    parser.add_argument('-d', '--dataset', default='training_dataset',
                        help='ava_v2.2')
    parser.add_argument('--pose', action='store_true', default=False, 
                        help='show 14 action pose of AVA.')
    parser.add_argument('--eval_split', default='val', type=str, 
                        help='set used to compute metrics')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_nano', type=str,
                        help='build YOWOv2')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")
    parser.add_argument('-e', '--epoch', default=20, type=int,
                        help='weight epoch')
    
    # tracking
    parser.add_argument('--min_duration', default=55, type=int,
                        help='')
    parser.add_argument('--decile_1', default=0, type=int,
                        help='')
    parser.add_argument('--decile_2', default=100, type=int, 
                        help='')
    parser.add_argument('--save_trajectory', action='store_true', default=True, 
                        help='')
    parser.add_argument('--correction_factor', default=0, type=int, 
                        help='')
    parser.add_argument('--max_age', default=5, type=int, 
                        help='')

    return parser.parse_args()
                    

@torch.no_grad()
def detect(len_clip, eval_split, dataset, nms_thresh, vis_thresh, model, device, transform, class_names, class_colors):

    path_evaluation = 'runs/evaluation/tracking'
    path_yowov2 = os.path.join(path_evaluation, 'yowov2')
    os.makedirs(path_yowov2, exist_ok=True)

    # list videos from the selected eval split 
    videos_to_analyze = []
    with open(f'data/{dataset}/splitfiles/{eval_split}list01.txt', 'r') as f:
        for line in f:
            extracted_part = os.path.basename(line.strip())
            videos_to_analyze.append(extracted_part)

    for video_name in videos_to_analyze:
        
        # load the video
        path_video = os.path.join('data/', dataset, 'videos', video_name+'.mp4')
        video = cv2.VideoCapture(path_video)

        # run
        video_clip = []
        index = 0
        yowov2_outputs = []

        while(True):
            ret, frame = video.read()
            
            if ret:
                # to RGB
                frame_rgb = frame[..., (2, 1, 0)]

                # to PIL image
                frame_pil = Image.fromarray(frame_rgb.astype(np.uint8))

                # prepare
                if len(video_clip) <= 0:
                    for _ in range(len_clip):
                        video_clip.append(frame_pil)

                video_clip.append(frame_pil)
                del video_clip[0]

                # orig size
                orig_h, orig_w = frame.shape[:2]

                # transform
                x, _ = transform(video_clip)
                # List [T, 3, H, W] -> [3, T, H, W]
                x = torch.stack(x, dim=1)
                x = x.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

                # inference
                outputs = model(x)

                # vis detection results
                batch_scores, batch_labels, batch_bboxes = outputs
                # batch size = 1
                scores = batch_scores[0]
                labels = batch_labels[0]
                bboxes = batch_bboxes[0]
                # rescale
                bboxes = rescale_bboxes(bboxes, [orig_w, orig_h])
                # one hot
                frame = vis_detection(
                    frame=frame,
                    scores=scores,
                    labels=labels,
                    bboxes=bboxes,
                    vis_thresh=vis_thresh,
                    class_names=class_names,
                    class_colors=class_colors
                    )
                
                # save bboxes
                for bbox, score in zip(bboxes, scores):
                        if score > nms_thresh:
                            yowov2_outputs.append({'frame': index,
                                                'x0': bbox[0],
                                                'y0': bbox[1],
                                                'x1': bbox[2],
                                                'y1': bbox[3],
                                                'confidence': score})
                            
            else:
                break

            index += 1
        
        # save results
        path_save_file = os.path.join(path_yowov2, video_name+'.csv')
        df_yowov2_outputs = pd.DataFrame(yowov2_outputs)
        df_yowov2_outputs.to_csv(path_save_file, index=False)

        video.release()
        cv2.destroyAllWindows()


def track(dataset, img_size, decile_1=0, decile_2=100, min_duration=55, save_trajectory=False, max_age=5):
    # create necessary folders
    for folder in ['tubes','trajectories']:
        path_evaluation = 'runs/evaluation/tracking/'
        directory = os.path.join(path_evaluation, folder)
        # if os.path.exists(directory):
        #     shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

    path_yowov2 = os.path.join(path_evaluation, 'yowov2')

    for video in os.listdir(path_yowov2):
        # load the video and associated yowov2 detections
        path_video = os.path.join('data', dataset, 'videos', os.path.splitext(video)[0]+'.mp4')
        path_detections = os.path.join(path_yowov2, video)
        df_yowov2_outputs = pd.read_csv(path_detections)
        # track bees with DeepSORT
        run_tubes = track_bees(df_yowov2_outputs, path_video, max_age)
        # thieve detections
        run_tubes_thieved = thieve_durations(run_tubes, min_duration)
        # get angles from runs
        run_tubes_angle = get_angles(run_tubes_thieved, img_size, os.path.splitext(video)[0], path_evaluation, decile_1, decile_2, save_trajectory)
        # save results
        run_tubes_angle.to_csv(os.path.join(path_evaluation, 'tubes', video), index=False)


def iou(row):
    x1, x2, y1, y2 =  row['x0'],row['x1'],row['y0'],row['y1']
    x3, x4, y3, y4 =  row['x0_d'],row['x1_d'],row['y0_d'],row['y1_d']

    if math.isnan(x1):
        return 0
    
    if math.isnan(x3):
        return 0

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    iou = inter_area / (area1 + area2 - inter_area)

    return iou


def match_gt_det(gt_df, detection_df):
    grouped = gt_df.groupby('run_id')
    pairs_iou = pd.DataFrame()

    # measurement of the iou for each detection / gt pairs
    for _, group in grouped: # iterate over gt dances
        run_id = int(group.head(1)['run_id'].iloc[0])
        for track_id in detection_df['run_id'].unique(): # iterate over detections
            track_df = detection_df[detection_df['run_id']==track_id]
            df_match = pd.merge(group[['frame_id','x0','x1','y0','y1']],
                                track_df[['frame_id','x0','x1','y0','y1']]
                                .rename(columns={col: f"{col}_d" for col in track_df.columns if col != 'frame_id'}),
                                on="frame_id", how="outer")
            df_match['iou'] = df_match.apply(iou, axis=1)
            row_to_add = pd.DataFrame({'GT':[run_id],
                                       'DET':[track_id],
                                       'IOU':[df_match['iou'].mean()],
                                       })
            pairs_iou = pd.concat([pairs_iou, row_to_add], ignore_index=True)

    if len(pairs_iou)==0:
        return pairs_iou
    
    # selection of pairs with the highest iou
    matches = pd.DataFrame()
    pairs_iou = pairs_iou.sort_values(by=['IOU'], ascending=[False])
    pairs_iou = pairs_iou.reset_index(drop=True)

    while len(pairs_iou)>0: # at each iteration, keep the pair with the highest iou and remove all other pairs with same detection / gt id
        row_to_add = pairs_iou.head(1)
        matches = pd.concat([matches, row_to_add], ignore_index=True)
        id_gt = pairs_iou.iloc[0]['GT']
        id_det = pairs_iou.iloc[0]['DET']
        pairs_iou = pairs_iou[(pairs_iou['GT'] != id_gt) & (pairs_iou['DET'] != id_det)]
    return matches


def angle_metric(gt_df, detection_df, matches, angles):
    if len(matches)==0:
        return angles
    
    for _, row in matches.iterrows():
        id_gt = row['GT']
        id_det = row['DET']
        gt_df = gt_df[gt_df['angle'] != 999]
        angle_gt = gt_df[gt_df['run_id']==id_gt]['angle'].mean()
        angle_det = detection_df[detection_df['run_id']==id_det]['angle'].mean()
        angles[0].append(angle_gt)
        angles[1].append(angle_det)

    return angles


def duration_metric(cf, gt_df, detection_df, matches, durations):
    for _, row in matches.iterrows():
        id_gt = row['GT']
        id_det = row['DET']
        frames_gt = list(gt_df[gt_df['run_id']==id_gt]['frame_id'])
        frames_det = list(detection_df[detection_df['run_id']==id_det]['frame_id'])
        durations[0].append(len(frames_gt))
        durations[1].append(len(frames_det)+cf)

    return durations


def eval(epoch, len_clip, version, correction_factor):
    # get annotations
    path_track = 'runs/evaluation/tracking/tubes'
    path_gt = 'data/training_dataset/annotations.txt'
    annotations = pd.read_csv(path_gt)

    # initiate lists
    angles = [[],[]]
    durations = [[],[]]
    nb_gt_dances=0
    nb_detected_dances=0

    # match detections and ground truth
    all_files = os.listdir(path_track)
    for file in all_files:
        detection_df = pd.read_csv(os.path.join(path_track,file))
        gt_df = annotations[annotations['filename']==file[:-4]]
        matches = match_gt_det(gt_df,detection_df)

        nb_gt_dances += len(gt_df['run_id'].unique())
        nb_detected_dances += len(matches)

        angles = angle_metric(gt_df,detection_df,matches,angles)
        durations = duration_metric(correction_factor,gt_df,detection_df,matches,durations)

    # metric function
    def calculate_metrics(true_values, predicted_values, metric_name):
        mae = mean_absolute_error(true_values, predicted_values)
        rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
        r2 = r2_score(true_values, predicted_values)
        bias = np.mean(np.array(true_values) - np.array(predicted_values))
        
        print(f'{metric_name} MAE: {mae}')
        print(f'{metric_name} RMSE: {rmse}')
        print(f'{metric_name} R2: {r2}')
        print(f'{metric_name} bias: {bias}')
        
        return mae, rmse, r2, bias

    # plot function
    def create_comparison_plot(true_values, predicted_values, title, x_label, y_label):
        max_val = max(max(true_values), max(predicted_values))
        min_val = min(min(true_values), min(predicted_values))
        
        fig = go.Figure(
            data=[
                go.Scatter(x=true_values, y=predicted_values, mode='markers', name=f'{title}', marker=dict(color='#2C3E50')),
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Identity line', line=dict(color='#E51400'))
            ],
            layout=go.Layout(
                title=title,
                xaxis=dict(title=x_label, mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey', zerolinewidth=1),
                yaxis=dict(title=y_label, scaleanchor="x", scaleratio=1, mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey', zeroline=True, zerolinecolor='lightgrey'),
                plot_bgcolor='white'
            )
        )
        
        r2 = r2_score(true_values, predicted_values)
        fig.update_layout(title=f'{title}, R2={round(r2, 2)}')
        return fig

    # main execution
    metrics = [
        ('angle', angles[0], angles[1]),
        ('duration', durations[0], durations[1])
    ]

    print(f'detected dances %: {(nb_detected_dances/nb_gt_dances)*100}')

    csv_save_path = os.path.join('runs/evaluation/tracking',"eval_track.csv")

    if os.path.exists(csv_save_path):
        df_metrics = pd.read_csv(csv_save_path)
    else:
        df_metrics = pd.DataFrame()

    to_add = pd.DataFrame({'version': [version],
                               'K': [len_clip],
                               'epoch': [epoch],
                               'detections_percentage': [(nb_detected_dances/nb_gt_dances)*100]})

    for name, true, pred in metrics:
        mae, rmse, r2, bias = calculate_metrics(true, pred, name)
        to_add[f'{name}_mae'] = round(mae,2)
        to_add[f'{name}_rmse'] = round(rmse, 2)
        to_add[f'{name}_r2'] = round(r2, 2)
        to_add[f'{name}_bias'] = round(bias, 2)
        
    df_metrics = pd.concat([df_metrics, to_add], ignore_index=True)
    df_metrics.to_csv(csv_save_path, index=False)

    for name, true, pred in metrics:        
        fig = create_comparison_plot(true, pred, f'Ground truth vs Predicted {name}s', f'Ground truth {name}s', f'Predicted {name}s')
        fig.write_image(f'runs/evaluation/tracking/{name}s_truevspred.png')


if __name__ == '__main__':
    np.random.seed(100)
    args = parse_args()

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    class_names = d_cfg['label_map']
    num_classes = d_cfg['valid_num_classes']

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # transform
    basetransform = BaseTransform(img_size=args.img_size)

    # build model
    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )

    # load trained weight
    version = args.version.split('_')[-1]
    path_to_ckpt = os.path.join('runs/training/weights', f'{version}_K{args.len_clip}', f'epoch_{args.epoch}.pth')
    model = load_weight(model=model, path_to_ckpt=path_to_ckpt)

    # to eval
    model = model.to(device).eval()

    # run
    detect(len_clip=args.len_clip,
           eval_split=args.eval_split,
           dataset=args.dataset,
           nms_thresh=args.nms_thresh,
           vis_thresh=args.vis_thresh,
           model=model,
           device=device,
           transform=basetransform,
           class_names=class_names,
           class_colors=class_colors)
    
    # run tracking
    track(dataset=args.dataset,
        img_size=args.img_size,
        decile_1=args.decile_1,
        decile_2=args.decile_2,
        min_duration=args.min_duration,
        save_trajectory=args.save_trajectory,
        max_age=args.max_age
        )

    # eval results
    print('------------------------------')
    eval(epoch=args.epoch,
        len_clip=args.len_clip,
        version=args.version,
        correction_factor=args.correction_factor
        )