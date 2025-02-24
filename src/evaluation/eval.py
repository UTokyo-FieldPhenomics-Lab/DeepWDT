from tqdm import tqdm

import numpy as np
import pandas as pd
from bytetracker import BYTETracker


def detect(df_results, dataloader, model, parameters, device):

    for iter_i, (frame_ids, video_clips, targets) in enumerate(tqdm(dataloader)):

        # Model inference
        video_clips = video_clips.to(device)
        outputs = model(video_clips) #scores, labels, boxes

        # Add results to a dataframe
        scores, labels, bboxes = outputs
        for index, (score, label, bboxe) in enumerate(zip(scores, labels, bboxes)):

            for detection_id in range(len(score)):

                to_add = {
                    'video': [targets[index]['video']],
                    'frame': [targets[index]['image_id']],
                    'class': [int(label[detection_id])],
                    'x0': [bboxe[detection_id][0]],
                    'y0': [bboxe[detection_id][1]],
                    'x1': [bboxe[detection_id][2]],
                    'y1': [bboxe[detection_id][3]],
                    'confidence': [score[detection_id]]
                }

                df_results = pd.concat([df_results, pd.DataFrame(to_add)], ignore_index=True)

    return df_results


def thieve_confidence(df, threshold=0):

    return df[df['confidence'] > threshold]


def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Each box is defined as [x0, y0, x1, y1].
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def nms(detections, threshold=0.5):
    """
    Remove non-necessary bounding boxes using non-maximum suppression (NMS).

    Args:
        detections (pandas.DataFrame): DataFrame containing detections with columns:
            [video, frame, class, x0, y0, x1, y1, confidence].
        threshold (float): IoU threshold to use for suppression.

    Returns:
        pandas.DataFrame: DataFrame with suppressed detections removed.
    """
    # List to hold the indices of detections we want to keep.
    keep_indices = []

    # Group the detections by video, frame, and class to apply NMS within each group.
    for group_keys, group in detections.groupby(['video', 'frame', 'class']):
        group = group.sort_values(by='confidence', ascending=False)

        boxes = group[['x0', 'y0', 'x1', 'y1']].values
        indices = group.index.tolist()

        suppressed = set()

        # Iterate over the detections in each group
        for i in range(len(boxes)):
            if indices[i] in suppressed:
                continue
            # Keep the current box.
            keep_indices.append(indices[i])
            # Compare this box with all the following boxes.
            for j in range(i + 1, len(boxes)):
                if indices[j] in suppressed:
                    continue
                # Compute the IoU between the current box and the j-th box.
                iou = compute_iou(boxes[i], boxes[j])
                # If the IoU exceeds the threshold, suppress the box.
                if iou > threshold:
                    suppressed.add(indices[j])

    # Return the detections corresponding to the kept indices.
    return detections.loc[keep_indices]


def track(df, threshold=0.3):
    """
    Produces a tube id for each detection by tracking detections in videos throughout frames using ByteTracker.

    Args:
        df (pandas.DataFrame): The dataframe containing YOWO detections
            with columns: [video, frame, class, x0, y0, x1, y1, confidence].
        threshold (float): IoU threshold for matching detections to tracker outputs.

    Returns:
        pandas.DataFrame: The original dataframe with an added column 'tube_id'
            indicating the track ID for each detection.
    """
    # Create a new column for tube_id and initialize with -1 (untracked)
    df = df.copy()
    df["tube_id"] = -1

    # Process each video separately (optionally you can also group by class)
    for video, video_df in df.groupby("video"):
        # Option: group by class if you want separate trackers per object category.
        # Here we assume objects of different classes should be tracked separately.
        for cls, group_df in video_df.groupby("class"):
            # Sort by frame to ensure correct temporal order.
            group_df = group_df.sort_values(by="frame")
            # Initialize a ByteTracker instance.
            # (Parameters here are examples; adjust track_thresh, match_thresh, track_buffer, and frame_rate as needed.)
            tracker = BYTETracker(track_thresh=threshold, match_thresh=0.8, track_buffer=30, frame_rate=30)

            # Process frame-by-frame.
            frames = group_df["frame"].unique()
            for frame in sorted(frames):
                frame_dets = group_df[group_df["frame"] == frame]

                # Prepare detections for ByteTracker.
                # ByteTracker expects detections as [x, y, w, h, score]
                dets = []
                for _, row in frame_dets.iterrows():
                    x, y = row["x0"], row["y0"]
                    w = row["x1"] - row["x0"]
                    h = row["y1"] - row["y0"]
                    score = row["confidence"]
                    dets.append([x, y, w, h, score])
                dets = np.array(dets) if dets else np.empty((0, 5))

                # Update tracker for the current frame.
                online_targets = tracker.update(dets, frame_id=int(frame))

                # For each detection in the frame, match with tracker output using IoU.
                for det_idx, (index, row) in enumerate(frame_dets.iterrows()):
                    # Detection bounding box in [x0, y0, x1, y1] format.
                    det_box = [row["x0"], row["y0"], row["x1"], row["y1"]]
                    best_iou = 0
                    best_track = -1
                    # Compare with each online target.
                    for target in online_targets:
                        # Tracker returns bounding box as tlwh: [x, y, w, h]
                        tx, ty, tw, th = target.tlwh
                        track_box = [tx, ty, tx + tw, ty + th]
                        iou_val = compute_iou(det_box, track_box)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_track = target.track_id
                    # If the best IoU exceeds the threshold, assign the tube_id.
                    if best_iou >= threshold:
                        df.loc[index, "tube_id"] = best_track

    return df


def get_metrics(df, tubes):
    return None


def eval_model(dataloader, tubes, model, parameters, device):
    df_results = pd.DataFrame(
        {
            'video': [],
            'frame': [],
            'class': [],
            'x0': [],
            'y0': [],
            'x1': [],
            'y1': [],
            'confidence': []
        }
    )

    detections = detect(df_results, dataloader, model, parameters, device)

    detections = thieve_confidence(detections)

    detection  = nms(detections)

    detections = track(detections)

    eval_metrics = get_metrics(detections, tubes)

    pass
