import argparse
import os
import torch

# from evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
# from evaluator.ava_evaluator import AVA_Evaluator
from src.evaluator.training_dataset_evaluator import TRAINING_DATASET_Evaluator

from src.dataset.transforms import BaseTransform

from src.utils.misc import load_weight, CollateFunc

from src.config import build_dataset_config, build_model_config
from src.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')

    # basic
    parser.add_argument('-bs', '--batch_size', default=8, type=int,
                        help='test batch size')
    parser.add_argument('-size', '--img_size', default=448, type=int,
                        help='the size of input frame')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_path', default='./evaluator/eval_results/',
                        type=str, help='Trained state_dict file path to open')

    # dataset
    parser.add_argument('-d', '--dataset', default='training_dataset',
                        help='ucf24, jhmdb, ava_v2.2.')
    parser.add_argument('--root', default='data/',
                        help='data root')

    # eval
    parser.add_argument('--cal_frame_mAP', action='store_true', default=False, 
                        help='calculate frame mAP.')
    parser.add_argument('--cal_video_mAP', action='store_true', default=False, 
                        help='calculate video mAP.')
    parser.add_argument('--eval_split', default='val', type=str, 
                        help='set used to compute metrics')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_large', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold. We suggest 0.005 for UCF24 and 0.1 for AVA.')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold. We suggest 0.5 for UCF24 and AVA.')
    parser.add_argument('--topk', default=40, type=int,
                        help='topk prediction candidates.')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('-m', '--memory', action="store_true", default=False,
                        help="memory propagate.")
    parser.add_argument('-e', '--epoch', default=20, type=int,
                        help='weight epoch')

    return parser.parse_args()


def eval(parameters, model, transform, collate_fn, metric, run_name):
    data_dir = os.path.join('data/', 'training_dataset')

    evaluator = TRAINING_DATASET_Evaluator(
        data_root=data_dir,
        dataset='training_dataset',
        model_name=parameters['MODEL_VERSION'],
        metric=metric,
        img_size=parameters['IMAGE_SIZE'],
        len_clip=parameters['LEN_CLIP'],
        batch_size=parameters['BATCH_SIZE']['EVAL'],
        conf_thresh=parameters['CONF_THRESH'],
        iou_thresh=parameters['IOU_THRESH'],
        transform=transform,
        collate_fn=collate_fn,
        gt_folder=parameters['GROUND_TRUTH'],
        eval_split=parameters['EVAL_SPLIT'],
        epoch=parameters['GROUND_TRUTH'],
        save_path=f'runs/eval/{run_name}'
    )

    if metric=='fmap':
        map = evaluator.evaluate_frame_map(model, run_name)

    elif metric=='vmap':
        map = evaluator.evaluate_video_map(model, run_name)

    return map


if __name__ == '__main__':
    args = parse_args()
    # dataset
    if args.dataset == 'training_dataset':
        num_classes = 1

    else:
        print('unknow dataset.')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

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
    if args.dataset == 'training_dataset':
        version = args.version.split('_')[-1]
        path_to_ckpt = os.path.join('runs/training/weights', f'{version}_K{args.len_clip}', f'epoch_{args.epoch}.pth')
        model = load_weight(model=model, path_to_ckpt=path_to_ckpt)
    else:
        model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    # transform
    basetransform = BaseTransform(img_size=args.img_size)

    # run
    if args.dataset == 'training_dataset':
        eval(
            args=args,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            collate_fn=CollateFunc(),
            metric='fmap'
            )