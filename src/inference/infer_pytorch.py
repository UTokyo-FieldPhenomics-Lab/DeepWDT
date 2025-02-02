import argparse
import json
import os
import time
import warnings

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from PIL import Image

# TO SOLVE LATER
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*DataFrameGroupBy.apply operated on the grouping columns.*"
)

from src.dataset.transforms import BaseTransform
from src.preprocessing.preprocessing import find_closer_32k
from src.postprocessing.postprocessing import to_labelme
from src.mapping.mapping import runs_2_loc, save_folium_map
from src.utils.misc import load_weight
from src.utils.box_ops import rescale_bboxes, rescale_bboxes_2
from src.utils.vis_tools import make_videos, vis_detection
from src.config import build_dataset_config, build_model_config
from src.models import build_model
from src.utils.tracking import get_angles, thieve_durations, track_bees


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2 Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=[448],
                        type=lambda x: [int(x)] if x.isdigit() else list(map(int, x.split(','))),
                        help='int or list')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('-vs', '--vis_thresh', default=0.3, type=float,
                        help='threshold for visualization')

    # class label config
    parser.add_argument('-d', '--dataset', default='ava_v2.2',
                        help='ava_v2.2')
    parser.add_argument('--video_format', default='mp4', type=str,
                        help='')

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
    parser.add_argument('--min_duration', default=20, type=int,
                        help='')
    parser.add_argument('--decile_1', default=0, type=int,
                        help='')
    parser.add_argument('--decile_2', default=100, type=int,
                        help='')
    parser.add_argument('--mapping', action="store_true", default=True,
                        help="memory propagate.")

    # others
    parser.add_argument('-method', '--translation_method', default='schurch', type=str,
                        help='')
    parser.add_argument('-cf', '--correction_factor', default=-14, type=int,
                        help='')
    parser.add_argument('--max_age', default=5, type=int,
                        help='')
    parser.add_argument('--ext_tool', default='labelme',
                        help='')
    parser.add_argument('--result_video', action='store_true', default=False,
                        help='')

    return parser.parse_args()


@torch.no_grad()
def detect(len_clip, dataset, nms_thresh, vis_thresh, model, device, transform, class_names, class_colors, new_size):
    path_save = os.path.join('runs/inference/', dataset, 'yowov2')
    os.makedirs(path_save, exist_ok=True)
    path_videos = os.path.join('data', dataset, 'videos')

    for video_name in os.listdir(path_videos):
        print(video_name)

        # path to video
        path_video = os.path.join(path_videos, video_name)

        # load video
        video = cv2.VideoCapture(path_video)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # run
        index = 0
        video_clip = []
        yowov2_outputs = []

        while (True):
            if index % 100 == 0:
                print(f'{index}/{total_frames}')
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
                x = x.unsqueeze(0).to(device)  # [B, 3, T, H, W], B=1

                t0 = time.time()
                # inference
                outputs = model(x)

                # vis detection results
                batch_scores, batch_labels, batch_bboxes = outputs
                # batch size = 1
                scores = batch_scores[0]
                labels = batch_labels[0]
                bboxes = batch_bboxes[0]
                # rescale
                if len(new_size) == 2:
                    bboxes = rescale_bboxes_2(bboxes, [orig_w, orig_h],
                                              new_size)  # yowo.py divide both dimension of bbox by max of frame width / height which is a problem for squared frames, so i used a custom rescale function
                else:
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
                                               'x0': round(bbox[0]),
                                               'y0': round(bbox[1]),
                                               'x1': round(bbox[2]),
                                               'y1': round(bbox[3]),
                                               'confidence': round(score, 2)})

            else:
                break

            index += 1

        # save results
        path_save_file = os.path.join(path_save, os.path.splitext(video_name)[0] + '.csv')
        df_yowov2_outputs = pd.DataFrame(yowov2_outputs)
        df_yowov2_outputs.to_csv(path_save_file, index=False)

        video.release()
        # cv2.destroyAllWindows()


def track(dataset, img_size, decile_1=0, decile_2=100, min_duration=20, max_age=5, video_format='mp4'):
    path_inference = 'runs/inference/'
    path_save = os.path.join(path_inference, dataset, 'tubes')
    os.makedirs(path_save, exist_ok=True)
    path_yowov2 = os.path.join(path_inference, dataset, 'yowov2')

    for video in os.listdir(path_yowov2):
        # load the video and associated yowov2 detections
        path_videos = os.path.join('data', dataset, 'videos', os.path.splitext(video)[0] + f'.{video_format}')
        path_detections = os.path.join(path_yowov2, video)
        df_yowov2_outputs = pd.read_csv(path_detections)
        # track bees with DeepSORT
        run_tubes = track_bees(df_yowov2_outputs, path_videos, max_age)
        # thieve detections
        run_tubes_thieved = thieve_durations(run_tubes, min_duration)
        # get angles from runs
        run_tubes_angle = get_angles(run_tubes_thieved, img_size, os.path.splitext(video)[0], path_inference, decile_1,
                                     decile_2, False)
        # save results
        run_tubes_angle.to_csv(os.path.join(path_inference, dataset, 'tubes', video), index=False)


def map_runs(dataset):
    path_tubes = os.path.join('runs/inference', dataset, 'tubes')
    path_geospatial = os.path.join('data', dataset, 'spatiotemporal_coordinates.json')
    path_videos = os.path.join('data', dataset, 'videos')
    os.makedirs(os.path.join('runs/inference', dataset, 'mapping'), exist_ok=True)
    output_geopackage = os.path.join('runs/inference', dataset, 'mapping/mapped_runs.gpkg')
    output_html_map = os.path.join('runs/inference', dataset, 'mapping/mapped_runs.html')

    # generate points data
    points_data = runs_2_loc(path_tubes, path_geospatial, path_videos)

    # save to geopackage
    gdf = gpd.GeoDataFrame(points_data, crs="EPSG:4326")
    gdf.to_file(output_geopackage, layer='mapped_runs', driver='GPKG')

    # save as a folium map
    save_folium_map(points_data, output_html_map)


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
    new_size = find_closer_32k(args.img_size)
    basetransform = BaseTransform(img_size=new_size)

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
    print('Detection ...')
    detect(len_clip=args.len_clip,
           dataset=args.dataset,
           nms_thresh=args.nms_thresh,
           vis_thresh=args.vis_thresh,
           model=model,
           device=device,
           transform=basetransform,
           class_names=class_names,
           class_colors=class_colors,
           new_size=new_size
           )
    print('Detection complete.')

    # run tracking
    print('Tracking ...')
    track(dataset=args.dataset,
          img_size=args.img_size,
          decile_1=args.decile_1,
          decile_2=args.decile_2,
          min_duration=args.min_duration,
          max_age=args.max_age,
          video_format=args.video_format
          )
    print('Tracking complete.')

    # make videos
    if args.result_video:
        print('Making result videos...')
        make_videos(args.dataset, args.video_format)
        print('Videos made.')

    # make external tool format annotations
    if args.ext_tool == 'labelme':
        print('Converting to labelme annotations...')
        to_labelme(args.dataset, video_format=args.video_format)
        print('Conversion made.')

    # map results
    if args.mapping:
        map_runs(dataset=args.dataset)

    print(f'Inference on {args.dataset} is complete.')