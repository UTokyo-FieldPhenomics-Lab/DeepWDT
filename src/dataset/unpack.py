import argparse
import cv2
import math
import os

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')
    parser.add_argument('--dataset', type=str, required=True)
    return parser.parse_args()

def unpack_frames(args):
    labels_dir = os.path.join('data', args.dataset, "rgb-images/Dancing")

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    video_filenames = [f for f in os.listdir(os.path.join('data', args.dataset, 'videos')) if f.endswith(('.mp4', '.avi'))]

    for video_filename in video_filenames:
        video_dir = os.path.join(labels_dir, os.path.splitext(video_filename)[0])
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        video_path = os.path.join(os.path.join('data', args.dataset, 'videos'), video_filename)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_filename = f"{frame_count:05d}.jpg"
            cv2.imwrite(os.path.join(video_dir, frame_filename), frame)

        cap.release()

def unpack_labels(args):
    df_train = pd.read_csv(os.path.join('data', args.dataset, 'train.csv'))
    df_val = pd.read_csv(os.path.join('data', args.dataset, 'val.csv'))
    df_test = pd.read_csv(os.path.join('data', args.dataset, 'test.csv'))
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    labels_dir = os.path.join('data', args.dataset, "labels/Dancing")

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    for filename in df['video'].unique():
        video_labels_dir = os.path.join(labels_dir, filename)
        if not os.path.exists(video_labels_dir):
            os.makedirs(video_labels_dir)

        video_df = df[df['video'] == filename]
        for frame_id in video_df['frame_id'].unique():
            frame_annotations = video_df[video_df['frame_id'] == frame_id]
            frame_filename = f"{int(frame_id):05d}.txt"

            with open(os.path.join(video_labels_dir, frame_filename), 'w') as file:
                for _, row in frame_annotations.iterrows():
                    annotation = f"1 {row['x0']} {row['y0']} {row['x1']} {row['y1']}\n"
                    file.write(annotation)

def check_videos(args):
    check_dir = os.path.join('data', args.dataset, "check")
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    df = pd.read_csv(os.path.join('data', args.dataset, 'train.csv'))
    video_filenames = [f for f in os.listdir(os.path.join('data', args.dataset, 'videos')) if f.endswith(('.mp4', '.avi'))]
    for video_filename in video_filenames:
        video_name = os.path.splitext(video_filename)[0]
        video_dir = os.path.join('data', args.dataset, "rgb-images/Dancing", video_name)
        frame_filenames = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        first_frame = cv2.imread(os.path.join(video_dir, frame_filenames[0]))
        frame_height, frame_width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(check_dir, f"{video_name}_annotated.mp4"), fourcc, 30, (frame_width, frame_height))
        video_df = df[df['video'] == video_name]
        grouped = video_df.groupby('frame_id')
        for frame_filename in frame_filenames:
            frame_id = int(os.path.splitext(frame_filename)[0])
            frame = cv2.imread(os.path.join(video_dir, frame_filename))
            if frame_id in grouped.groups:
                for _, row in grouped.get_group(frame_id).iterrows():
                    x0, y0, x1, y1 = int(row['x0']), int(row['y0']), int(row['x1']), int(row['y1'])
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    length = 30
                    angle = row['angle']
                    dx = int(length * math.sin(angle))
                    dy = int(-length * math.cos(angle))
                    cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (0, 0, 255), 2)
            out.write(frame)
        out.release()

if __name__ == '__main__':
    print('Preparing dataset...')
    args = parse_args()
    unpack_frames(args)
    unpack_labels(args)
    check_videos(args)
    print('Dataset ready!')
