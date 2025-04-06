import math
from functools import partial

import numpy as np
from sklearn.decomposition import PCA


def correct_angle(angle, df_id):
    if angle < 0:
        angle += math.pi

    if 0 <= angle < (1 / 4) * math.pi:
        x_diff = df_id.iloc[-1]['center_x'] - df_id.iloc[0]['center_x']
        angle += math.pi / 2 if x_diff > 0 else -math.pi / 2

    elif (1 / 4) * math.pi <= angle < (1 / 2) * math.pi:
        y_diff = df_id.iloc[-1]['center_y'] - df_id.iloc[0]['center_y']
        angle += math.pi / 2 if y_diff > 0 else -math.pi / 2

    elif math.pi / 2 <= angle < (3 / 4) * math.pi:
        y_diff = df_id.iloc[-1]['center_y'] - df_id.iloc[0]['center_y']
        angle += -math.pi / 2 - math.pi if y_diff > 0 else -math.pi / 2

    else:
        x_diff = df_id.iloc[-1]['center_x'] - df_id.iloc[0]['center_x']
        angle += -3 * math.pi / 2 if x_diff <= 0 else -math.pi / 2

    return angle


def thieve_deciles(group, decile_1=0, decile_2=100):
    """
        filtering detections between two defined deciles within runs
    """
    n = len(group)
    start = int(n * decile_1)
    end = int(n * decile_2)
    return group.iloc[start:end]


def get_angles(detections, decile_1=0, decile_2=100):
    """
        Run a PCA to find run direction vectors.
    """

    detections['center_x'], detections['center_y'] = ((detections['x0'] + detections['x1']) / 2,
                                                      (detections['y0'] + detections['y1']) / 2)

    detections = (detections.sort_values(by=['run_id', 'frame_id'])
                            .groupby('run_id', group_keys=False)
                            .apply(partial(thieve_deciles, decile_1=decile_1, decile_2=decile_2)))

    angle_dict = {}

    for id in detections['run_id'].unique():
        df_id = detections[detections['run_id'] == id].sort_values(by='frame_id')

        pca = PCA(n_components=1)
        pca.fit(df_id[['center_x', 'center_y']])
        PC1 = pca.components_[0]
        angle = np.arctan2(PC1[1], PC1[0])

        angle_dict[id] = correct_angle(angle, df_id)

    detections['angle'] = detections['run_id'].map(angle_dict)

    return detections.drop(['center_x', 'center_y'], axis=1)