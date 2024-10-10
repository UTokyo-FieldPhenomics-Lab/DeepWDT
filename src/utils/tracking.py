import numpy as np

import cv2
import pandas as pd
from functools import partial
from sklearn.decomposition import PCA
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.utils.angles import correct_angle
from src.utils.vis_tools import trajectory_visualization


def thieve_durations(df, min_duration=55):
    """
        thieving runs based on the durations
    """
    group_counts = df.groupby('run_id')['frame_id'].transform('count')
    return df[group_counts >= min_duration]


def thieve_deciles(group, decile_1=0, decile_2=100):
    """
        filtering detections between two defined deciles within runs
    """
    n = len(group)
    start = int(n * decile_1)
    end = int(n * decile_2)
    return group.iloc[start:end]


def get_angles(run_tubes_thieved, img_size, video_name, path_evaluation, decile_1, decile_2, save_trajectory):
    """
        pca to find run direction vectors
    """
    run_tubes_thieved_wd = run_tubes_thieved.copy()

    run_tubes_thieved_wd['center_x'], run_tubes_thieved_wd['center_y'] = (run_tubes_thieved_wd['x0'] + run_tubes_thieved_wd['x1']) / 2, (run_tubes_thieved_wd['y0'] + run_tubes_thieved_wd['y1']) / 2
    
    run_tubes_thieved_wd = (run_tubes_thieved_wd.sort_values(by=['run_id', 'frame_id'])
              .groupby('run_id', group_keys=False)
              .apply(partial(thieve_deciles, decile_1=decile_1, decile_2=decile_2)))

    angle_dict = {}

    for id in run_tubes_thieved_wd['run_id'].unique():
        df_id = run_tubes_thieved_wd[run_tubes_thieved_wd['run_id'] == id].sort_values(by='frame_id')

        pca = PCA(n_components=1)
        pca.fit(df_id[['center_x', 'center_y']])
        PC1 = pca.components_[0]
        angle = np.arctan2(PC1[1], PC1[0])
        angle_dict[id] = correct_angle(angle, df_id)

        if save_trajectory:
            filename = video_name + '_' + str(id) + '.png'
            trajectory_visualization(angle, df_id, filename, img_size, path_evaluation)

    run_tubes_thieved_wd['angle'] = run_tubes_thieved_wd['run_id'].map(angle_dict)

    return run_tubes_thieved_wd.drop(['center_x', 'center_y'], axis=1)


def track_bees(df_yowov2_outputs, path_videos, max_age=5):
    """
    linking individual detections to make tubes
    """
    cap = cv2.VideoCapture(path_videos)
    tracker = DeepSort(max_age=max_age)
    tracking_data = []
    frame_id = 0

    df_yowov2_outputs['w'] = df_yowov2_outputs['x1'] - df_yowov2_outputs['x0']
    df_yowov2_outputs['h'] = df_yowov2_outputs['y1'] - df_yowov2_outputs['y0']

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_view = (df_yowov2_outputs[df_yowov2_outputs['frame'] == frame_id][['x0','y0','w','h','confidence']]
        .assign(constant=1)
        .values
        .tolist())

        bbs = [[sublist[:4], *sublist[4:]] for sublist in frame_view]

        tracks = tracker.update_tracks(bbs, frame=frame)

        for track in tracks:
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = bbox
            tracking_data.append({
                'x0': x1,
                'y0': y1,
                'x1': x2,
                'y1': y2,
                'run_id': track_id,
                'frame_id': frame_id
            })

        frame_id += 1

    cap.release()

    df_tracking = pd.DataFrame(tracking_data)

    return df_tracking