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


def ucf_jhmdb_eval(args, d_cfg, model, transform, collate_fn):
    data_dir = os.path.join(args.root, 'ucf24')
    if args.cal_frame_mAP:
        # Frame mAP evaluator
        evaluator = UCF_JHMDB_Evaluator(
            data_root=data_dir,
            dataset=args.dataset,
            model_name=args.version,
            metric='fmap',
            img_size=args.img_size,
            len_clip=args.len_clip,
            batch_size=args.batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            gt_folder=d_cfg['gt_folder'],
            save_path=args.save_path
            )
        # evaluate
        evaluator.evaluate_frame_map(model, show_pr_curve=True)

    elif args.cal_video_mAP:
        # Video mAP evaluator
        evaluator = UCF_JHMDB_Evaluator(
            data_root=data_dir,
            dataset=args.dataset,
            model_name=args.version,
            metric='vmap',
            img_size=args.img_size,
            len_clip=args.len_clip,
            batch_size=args.batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            gt_folder=d_cfg['gt_folder'],
            save_path=args.save_path
            )
        # evaluate
        evaluator.evaluate_video_map(model)


def ava_eval(args, d_cfg, model, transform, collate_fn):
    data_dir = os.path.join(args.root, 'AVA_Dataset')
    evaluator = AVA_Evaluator(
        d_cfg=d_cfg,
        data_root=data_dir,
        img_size=args.img_size,
        len_clip=args.len_clip,
        sampling_rate=d_cfg['sampling_rate'],
        batch_size=args.batch_size,
        transform=transform,
        collate_fn=collate_fn,
        full_test_on_val=False,
        version='v2.2')

    mAP = evaluator.evaluate_frame_map(model)


def training_dataset_eval(args, d_cfg, model, transform, collate_fn):
    data_dir = os.path.join(args.root, 'training_dataset')
    if args.cal_frame_mAP:
        # Frame mAP evaluator
        evaluator = TRAINING_DATASET_Evaluator(
            data_root=data_dir,
            dataset=args.dataset,
            model_name=args.version,
            metric='fmap',
            img_size=args.img_size,
            len_clip=args.len_clip,
            batch_size=args.batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            gt_folder=d_cfg['gt_folder'],
            eval_split=args.eval_split,
            epoch=args.epoch,
            save_path=args.save_path
            )
        # evaluate
        evaluator.evaluate_frame_map(model, show_pr_curve=True)

    elif args.cal_video_mAP:
        # Video mAP evaluator
        evaluator = TRAINING_DATASET_Evaluator(
            data_root=data_dir,
            dataset=args.dataset,
            model_name=args.version,
            metric='vmap',
            img_size=args.img_size,
            len_clip=args.len_clip,
            batch_size=args.batch_size,
            conf_thresh=0.01,
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            gt_folder=d_cfg['gt_folder'],
            eval_split=args.eval_split,
            epoch=args.epoch,
            save_path=args.save_path
            )
        # evaluate
        evaluator.evaluate_video_map(model)


if __name__ == '__main__':
    args = parse_args()
    # dataset
    if args.dataset == 'ucf24':
        num_classes = 24

    elif args.dataset == 'jhmdb':
        num_classes = 21

    elif args.dataset == 'ava_v2.2':
        num_classes = 80

    elif args.dataset == 'training_dataset':
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
    if args.dataset in ['ucf24', 'jhmdb21']:
        ucf_jhmdb_eval(
            args=args,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            collate_fn=CollateFunc()
            )
    elif args.dataset == 'ava_v2.2':
        ava_eval(
            args=args,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            collate_fn=CollateFunc()
            )
    elif args.dataset == 'training_dataset':
        training_dataset_eval(
            args=args,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            collate_fn=CollateFunc()
            )