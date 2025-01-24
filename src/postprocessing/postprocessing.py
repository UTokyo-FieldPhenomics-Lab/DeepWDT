import json
import os

import cv2
import pandas as pd


def to_labelme(dataset, video_format):
    path_tubes = os.path.join('runs/inference', dataset, 'tubes')
    path_labelme = os.path.join('runs/inference', dataset, 'labelme')
    path_videos = os.path.join('data', dataset, 'videos')
    os.makedirs(path_labelme, exist_ok=True)

    for tube_file in os.listdir(path_tubes):
        tubes = pd.read_csv(os.path.join(path_tubes, tube_file))

        video_name = os.path.splitext(tube_file)[0] + f'.{video_format}'
        video_path = os.path.join(path_videos, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video {video_name}")
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_data = tubes[tubes['frame_id'] == frame_count]
            if not frame_data.empty:
                labelme_annotation = {
                    "version": "4.5.6",
                    "flags": {},
                    "shapes": [],
                    "imagePath": f"{frame_count:05d}.jpg",
                    "imageData": None,
                    "imageHeight": frame.shape[0],
                    "imageWidth": frame.shape[1]
                }

                for _, row in frame_data.iterrows():
                    points = [
                        [row['x0'], row['y0']],
                        [row['x1'], row['y0']],
                        [row['x1'], row['y1']],
                        [row['x0'], row['y1']]
                    ]

                    shape = {
                        "label": "dancing",
                        "points": points,
                        "group_id": int(row['run_id']),
                        "shape_type": "polygon",
                        "flags": {},
                        "angle": float(row['angle'])
                    }

                    labelme_annotation["shapes"].append(shape)

                video_frame_dir = os.path.join(path_labelme, os.path.splitext(video_name)[0])
                os.makedirs(video_frame_dir, exist_ok=True)

                json_path = os.path.join(video_frame_dir, f"{frame_count:05d}.json")
                with open(json_path, 'w') as json_file:
                    json.dump(labelme_annotation, json_file, indent=4)

                frame_image_path = os.path.join(video_frame_dir, f"{frame_count:05d}.jpg")
                cv2.imwrite(frame_image_path, frame)

            frame_count += 1

        cap.release()
