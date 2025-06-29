import numpy as np
import pandas as pd

from .sort import Sort, iou_batch
from .angles import get_angles


def track(df, iou_threshold=0.3, duration_threshold=15, max_age=5):
    """
    Produces a tube id for each detection by tracking detections in videos
    throughout frames using SORT.

    Args:
        df (pandas.DataFrame): The dataframe containing YOWO detections
            with columns: [video, frame, class, x0, y0, x1, y1, confidence].
        threshold (float): IoU threshold for matching detections to tracker outputs.

    Returns:
        pandas.DataFrame: The original dataframe with an added column 'run_id'
            indicating the track ID for each detection.
    """

    # Make sure we have a copy of the dataframe to avoid modifying the original.
    df = df.copy()
    # Prepare a list to hold per-video results
    results = []

    # Process each video separately
    for video_id in df['video'].unique():
        video_df = df[df['video'] == video_id].copy()
        video_df.sort_values(by='frame_id', inplace=True)
        # Initialize a SORT tracker for this video
        tracker = Sort(max_age=3, iou_threshold=iou_threshold)
        # To collect processed frames for the video
        video_results = []

        # Process frames in order
        for frame in sorted(video_df['frame_id'].unique()):
            frame_df = video_df[video_df['frame_id'] == frame].copy()
            # Prepare detections: array of [x0, y0, x1, y1, confidence]
            dets = frame_df[['x0', 'y0', 'x1', 'y1', 'confidence']].values

            # Update the tracker for this frame.
            tracks = tracker.update(dets)

            # For each detection, we compute the IoU with every track output.
            if tracks.shape[0] > 0:
                # Compute IoU between detections (as bb_test) and tracker boxes (as bb_gt)
                ious = iou_batch(frame_df[['x0', 'y0', 'x1', 'y1']].values, tracks[:, :4])
                # For each detection, pick the tracker with the highest IoU.
                max_ious = ious.max(axis=1)
                best_match_idx = ious.argmax(axis=1)
                # Assign run_id if IoU exceeds threshold; otherwise assign NaN.
                run_ids = [
                    tracks[best_match_idx[i], 4] if max_ious[i] >= iou_threshold else np.nan
                    for i in range(len(max_ious))
                ]
                frame_df['run_id'] = run_ids
            else:
                frame_df['run_id'] = np.nan

            video_results.append(frame_df)

        results.append(pd.concat(video_results, ignore_index=True))

    if not results:
        return pd.DataFrame({})

    df_results = pd.concat(results, ignore_index=True)

    df_results = thieve_durations(df_results, threshold=duration_threshold)

    df_results = get_angles(df_results)

    df_results['run_id'] = df_results['run_id'].astype(int)

    return df_results


def thieve_durations(detections, threshold):
    """
     Thieves runs based on the durations.
    """

    group_counts = detections.groupby('run_id')['frame_id'].transform('count')

    return detections[group_counts >= threshold]