import os
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
from .sort import Sort, iou_batch


def detect(df_results, dataloader, model, parameters, device):

    for iter_i, (frame_ids, video_clips, targets) in enumerate(tqdm(dataloader)):

        # Model inference
        video_clips = video_clips.to(device)
        outputs = model(video_clips) # scores, labels, bboxes

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


def thieve_confidence(df, threshold=0.1):

    return df[df['confidence'] >= threshold]


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
    Produces a tube id for each detection by tracking detections in videos
    throughout frames using SORT.

    Args:
        df (pandas.DataFrame): The dataframe containing YOWO detections
            with columns: [video, frame, class, x0, y0, x1, y1, confidence].
        threshold (float): IoU threshold for matching detections to tracker outputs.

    Returns:
        pandas.DataFrame: The original dataframe with an added column 'tube_id'
            indicating the track ID for each detection.
    """

    # Make sure we have a copy of the dataframe to avoid modifying the original.
    df = df.copy()
    # Prepare a list to hold per-video results
    results = []

    # Process each video separately
    for video_id in df['video'].unique():
        video_df = df[df['video'] == video_id].copy()
        video_df.sort_values(by='frame', inplace=True)
        # Initialize a SORT tracker for this video
        tracker = Sort(max_age=2, iou_threshold=threshold)
        # To collect processed frames for the video
        video_results = []

        # Process frames in order
        for frame in sorted(video_df['frame'].unique()):
            frame_df = video_df[video_df['frame'] == frame].copy()
            # Prepare detections: array of [x0, y0, x1, y1, confidence]
            dets = frame_df[['x0', 'y0', 'x1', 'y1', 'confidence']].values

            # Update the tracker for this frame.
            # The update method returns an array with rows of [x0, y0, x1, y1, tube_id]
            tracks = tracker.update(dets)

            # For each detection, we compute the IoU with every track output.
            # Use the provided iou_batch function (assumed to be in scope).
            if tracks.shape[0] > 0:
                # Compute IoU between detections (as bb_test) and tracker boxes (as bb_gt)
                ious = iou_batch(frame_df[['x0', 'y0', 'x1', 'y1']].values, tracks[:, :4])
                # For each detection, pick the tracker with the highest IoU.
                max_ious = ious.max(axis=1)
                best_match_idx = ious.argmax(axis=1)
                # Assign tube_id if IoU exceeds threshold; otherwise assign NaN.
                tube_ids = [
                    tracks[best_match_idx[i], 4] if max_ious[i] >= threshold else np.nan
                    for i in range(len(max_ious))
                ]
                frame_df['tube_id'] = tube_ids
            else:
                frame_df['tube_id'] = np.nan

            video_results.append(frame_df)
        results.append(pd.concat(video_results, ignore_index=True))
    return pd.concat(results, ignore_index=True)


def thieve_length(detections, threshold=0):
    pass


def visualization(detections):
    """
    Save videos with results plot on it.

    Args:
        detections (pandas.DataFrame): The dataframe containing YOWO detections
            with columns: [video (without extension, need to add mp4), frame, class,
            x0, y0, x1, y1, confidence, tube_id].
    """
    # Define paths.
    path_videos = '/Users/sylvaingrs/Documents/Github/DeepWDT/data/bengaluru_01/videos'
    path_save = '/Users/sylvaingrs/Documents/Github/DeepWDT/runs/test/'

    # Ensure the save directory exists.
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    # Process each unique video.
    for video in detections['video'].unique():
        video_file = os.path.join(path_videos, video + '.mp4')
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue

        # Get video properties.
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_file = os.path.join(path_save, video + '_vis.mp4')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

        # Filter detections for the current video.
        video_dets = detections[detections['video'] == video]

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections corresponding to the current frame.
            current_dets = video_dets[video_dets['frame'] == frame_idx]

            # Draw each detection on the frame.
            for _, row in current_dets.iterrows():
                x0, y0, x1, y1 = int(row['x0']), int(row['y0']), int(row['x1']), int(row['y1'])
                conf = row['confidence']
                cls = row['class']
                # Draw bounding box.
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                # Prepare label: if tube_id is nan, show "NA"
                label = f"ID:{cls} {conf:.2f}"
                # Put label above the bounding box.
                cv2.putText(frame, label, (x0, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write the frame with overlayed detections.
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Saved visualization video to {out_file}")


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

    print('1. Detecting dancing bees...')
    detections = detect(df_results, dataloader, model, parameters, device)

    print('2. Thieving detections based on the confidence score...')
    detections = thieve_confidence(detections)
    detections[['x0', 'x1', 'y0', 'y1']] = detections[['x0', 'x1', 'y0', 'y1']] * 224
    visualization(detections)

    print('3. Running NMS...')
    detections  = nms(detections)

    print('4. Tracking dancing bees...')

    detections = track(detections)

    # print('5. Thieving short dances...')
    # detections = thieve_length(detections)

    visualization(detections)

    print('6. Computing evaluation metrics...')
    eval_metrics = get_metrics(detections, tubes)

    pass
