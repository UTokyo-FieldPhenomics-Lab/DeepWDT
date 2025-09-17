import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Each box is defined as a tuple: (x0, y0, x1, y1).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def match_gt_det(detection, gt_tubes):
    """
    Match ground truth tubes with the best detected one.

    Args:
        detection (pandas.DataFrame): DataFrame containing detections with columns:
            [video, frame_id, x0, y0, x1, y1, confidence, run_id].
        gt_tubes (pandas.DataFrame): DataFrame containing ground truth tubes with columns:
            [video, frame_id, x0, y0, x1, y1, run_id].

    Returns:
        matches (pandas.DataFrame): DataFrame with matched run_id from detection and gt_tubes.
            Columns include: video, gt_run_id, det_run_id, and the matching score.
    """
    matches = []

    # Process tube matching for each video
    for video in gt_tubes['video'].unique():
        gt_video = gt_tubes[gt_tubes['video'] == video]
        det_video = detection[detection['video'] == video]

        # Group tubes by run_id in ground truth and detection dataframes
        gt_groups = gt_video.groupby('run_id')
        det_groups = det_video.groupby('run_id')

        # Iterate over ground truth runs
        for gt_run, gt_group in gt_groups:
            best_score = 0
            best_det_run = None

            # Iterate over detected runs
            for det_run, det_group in det_groups:
                scores = []

                # For each frame in the ground truth tube, try to compute IoU
                for _, gt_row in gt_group.iterrows():
                    frame = gt_row['frame_id']
                    # Find detection boxes in the same frame
                    det_boxes = det_group[det_group['frame_id'] == frame]
                    if not det_boxes.empty:
                        # Compute IoU for each detection in this frame and pick the highest value
                        ious = []
                        for _, det_row in det_boxes.iterrows():
                            iou = compute_iou(
                                (gt_row['x0'], gt_row['y0'], gt_row['x1'], gt_row['y1']),
                                (det_row['x0'], det_row['y0'], det_row['x1'], det_row['y1'])
                            )
                            ious.append(iou)
                        max_iou = max(ious) if ious else 0
                        scores.append(max_iou)

                # Compute average IoU over all overlapping frames
                avg_iou = sum(scores) / len(scores) if scores else 0

                if avg_iou > best_score:
                    best_score = avg_iou
                    best_det_run = det_run

            # Keep the best matching
            matches.append({
                'video': video,
                'gt_run_id': gt_run,
                'detection_run_id': best_det_run,
                'average_iou': best_score
            })

    matches_df = pd.DataFrame(matches)

    matches_df = matches_df.dropna(subset=['detection_run_id'])

    return matches_df


def match_angles(detections, gt_tubes, matches):
    angles = [[], []]

    for _, row in matches.iterrows():
        id_gt = row['gt_run_id']
        id_det = row['detection_run_id']
        video = row['video']

        gt_tubes = gt_tubes[gt_tubes['angle'] != 999]

        angle_gt = gt_tubes[(gt_tubes['video'] == video) & (gt_tubes['run_id'] == id_gt)]['angle'].mean()
        angle_det = detections[(detections['video'] == video) & (detections['run_id'] == id_det)]['angle'].mean()

        angles[0].append(angle_gt)
        angles[1].append(angle_det)

    return angles


def match_durations(detections, gt_tubes, matches, cf=0):
    durations = [[], []]

    for _, row in matches.iterrows():
        id_gt = row['gt_run_id']
        id_det = row['detection_run_id']
        video = row['video']

        frames_gt = list(gt_tubes[(gt_tubes['video'] == video) & (gt_tubes['run_id'] == id_gt)]['frame_id'])
        frames_det = list(detections[(detections['video'] == video) & (detections['run_id'] == id_det)]['frame_id'])

        durations[0].append(len(frames_gt))

        set_gt = set(frames_gt)
        set_dt = set(frames_det)
        durations[1].append(len(set_gt.intersection(set_dt)) + cf)

    return durations


def get_metrics(gt_tubes, detections, cf):
    matches = match_gt_det(detections, gt_tubes)

    angles = match_angles(detections, gt_tubes, matches)
    if not angles[0]:
        angle_rmse = -1
        angle_r2 = -1
    else:
        angle_rmse = np.sqrt(mean_squared_error(angles[0], angles[1]))
        angle_r2 = r2_score(angles[0], angles[1]) # (ground truth, detected)

    durations = match_durations(detections, gt_tubes, matches, cf=cf)
    if not durations[0]:
        duration_rmse = -1
        mean_duration_error = -1
        stdv_duration_error = -1
        r_value_duration_pearson = -1
        p_value_duration_pearson = -1
        duration_r2 = -1
    else:
        duration_rmse = np.sqrt(mean_squared_error(durations[0], durations[1]))
        duration_errors = [a - b for a, b in zip(durations[0], durations[1])]
        mean_duration_error = np.mean(duration_errors)
        stdv_duration_error = np.std(duration_errors)
        r_value_duration_pearson, p_value_duration_pearson = pearsonr(durations[0], duration_errors)
        duration_r2 = r2_score(durations[0], durations[1])  # (ground truth, detected)

    # Percentage of detected runs (is equal to recall)
    detected_runs = (len(matches) / gt_tubes.groupby(['video', 'run_id']).ngroups) * 100

    # Precision and recall
    nb_detections = detections.groupby(['video', 'run_id']).ngroups
    nb_gt = gt_tubes.groupby(['video', 'run_id']).ngroups
    true_positives = len(matches)
    false_positives = nb_detections - true_positives
    false_negatives = nb_gt - true_positives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # Number of detected runs per dance
    dance_groups_det = matches.groupby(matches['video'].str[:-3])['gt_run_id'].nunique().to_dict()
    dance_groups_gt = gt_tubes.groupby(gt_tubes['video'].str[:-3])['run_id'].nunique().to_dict()
    nb_detected_runs_per_dance = np.mean([dance_groups_det.get(dance, 0) / count for dance, count in dance_groups_gt.items()])

    return ({'angle_rmse': round(angle_rmse, 2),
             'duration_rmse': round(duration_rmse, 2),
             'mean_duration_error': round(mean_duration_error, 2),
             'stdv_duration_error': round(stdv_duration_error, 2),
             'detected_runs': round(detected_runs, 1),
             'r_value_duration_pearson': round(r_value_duration_pearson, 2),
             'p_value_duration_pearson': round(p_value_duration_pearson, 2),
             'angle_r2': round(angle_r2, 2),
             'duration_r2': round(duration_r2, 2),
             'precision': round(precision,2),
             'recall': round(recall, 2),
             'nb_detected_runs_per_dance': round(nb_detected_runs_per_dance, 2)},
            angles,
            durations,
            matches)
