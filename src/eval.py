import os
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from src.dataset import build_dataset, CollateFunction, EvalTransform
from src.evaluation import get_metrics
from src.model import build_yowo_model, track
from src.utils import thieve_confidence, grouped_nms, make_evaluation_graphs, make_evaluation_trajectory_graphs
from src.config import load_configuration


def eval_function(path_configuration):
    configuration = load_configuration(path_configuration)
    eval_configuration = configuration.eval

    print("Arguments: ", configuration)
    print('-----------------------------------------------------------------------------------------------------------')

    # Instantiate the validation dataset and dataloader
    print('Loading the validation dataset...')
    validation_dataset = build_dataset(
        configuration=eval_configuration.dataset,
        transform=EvalTransform(img_size=eval_configuration.dataset.image_size[0]),
        split=eval_configuration.split
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
    device = eval_configuration.device
    model = build_yowo_model(model_configuration=eval_configuration.model,
                             device=device,
                             trainable=True,
                             centered_clip=eval_configuration.dataset.centered_clip)
    checkpoint_path = torch.load(eval_configuration.weights, map_location='cpu').pop("model")
    model.load_state_dict(checkpoint_path)
    model = model.to(device).eval()

    model.trainable = False
    print('Model built!')

    # Evaluate the model
    print('Evaluating the model...')
    model.eval()

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
                       iou_threshold=eval_configuration.track_iou_threshold,
                       duration_threshold=eval_configuration.track_duration_threshold)

    # Compute evaluation metrics
    print('3. Computing evaluation metrics...')
    if len(detections) == 0:
        print(f'[Detected runs: 0%]')
    else:
        detections[['x0', 'x1', 'y0', 'y1']] = detections[['x0', 'x1', 'y0', 'y1']] * eval_configuration.dataset.image_size[0]
        eval_metrics, angles, durations, matches = get_metrics(validation_dataset.df, detections, eval_configuration.correction_factor)

        # Print metrics
        print(
            f'[Detected runs: {eval_metrics["detected_runs"]}%][Angle RMSE: {eval_metrics["angle_rmse"]}][Duration RMSE: {eval_metrics["duration_rmse"]}]')
        print(f'Angle R²: {eval_metrics["angle_r2"]:.3f}')
        print(f'Duration R²: {eval_metrics["duration_r2"]:.3f}')
        print(f'Precision: {eval_metrics["precision"]:.3f}')
        print(f'Recall: {eval_metrics["recall"]:.3f}')
        print(f'Mean duration error: {eval_metrics["mean_duration_error"]:.3f}')
        print(f'Std duration error: {eval_metrics["stdv_duration_error"]:.3f}')
        print(
            f'Pearson correlation (duration): {eval_metrics["r_value_duration_pearson"]:.3f} (p={eval_metrics["p_value_duration_pearson"]:.3f})')
        print(f'Detected runs per dance: {eval_metrics["nb_detected_runs_per_dance"]:.3f}')

        # Save graphs
        weights_path = Path(eval_configuration.weights)
        run_name = weights_path.parent.name if weights_path.parent.name != 'weights' else weights_path.parent.parent.name
        path_graphs = Path(f'runs/eval/{run_name}/graphs')
        os.makedirs(path_graphs, exist_ok=True)
        path_graphs_gtvsp_angles = path_graphs / 'gt_vs_predicted_angles.png'
        path_graphs_gtvsp_durations = path_graphs / 'gt_vs_predicted_durations.png'
        path_graphs_duration_error_vs_duration = path_graphs / 'duration_errors_vs_durations.png'
        make_evaluation_graphs(angles,
                               durations,
                               path_graphs_gtvsp_angles,
                               path_graphs_gtvsp_durations,
                               path_graphs_duration_error_vs_duration,
                               eval_metrics['angle_r2'],
                               eval_metrics['duration_r2'])
        trajectory_graphs_path = path_graphs / 'trajectories'
        make_evaluation_trajectory_graphs(validation_dataset.df, detections, matches, trajectory_graphs_path, eval_configuration.dataset.image_size[0])
