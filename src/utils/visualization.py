import os
import cv2
import random
from pathlib import Path

import pandas as pd
import numpy as np

def make_video(detections, run_path):
    """
    Create videos with visually identified dancing bees.

    Args:
        detections (pandas.DataFrame): DataFrame with columns: [video, frame_id, run_id, class, x0, y0, x1, y1, confidence].
        run_path (str): Directory used to save visualizations.
    """
    # Define the color palette
    color_palette = [
        (3, 4, 94),    # 03045E
        (2, 62, 138),   # 023E8A
        (0, 119, 182)   # 0077B6
    ]

    # Group detections by video
    grouped = detections.groupby('video')

    for video_name, video_detections in grouped:
        # Create a directory for the video if it doesn't exist
        video_dir = run_path / Path(video_name).stem
        os.makedirs(video_dir, exist_ok=True)

        # Define the output video path
        output_video_path = os.path.join(video_dir, Path(video_name).name)
        print(output_video_path)

        # Open the input video file
        input_video_path = os.path.join(video_name)
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections for the current frame
            frame_detections = video_detections[video_detections['frame_id'] == frame_id]

            # Overlay bounding boxes on the frame
            for _, detection in frame_detections.iterrows():
                x0, y0, x1, y1 = int(detection['x0']), int(detection['y0']), int(detection['x1']), int(detection['y1'])
                run_id = detection['run_id']

                # Pick a random color from the palette
                color = random.choice(color_palette)

                # Draw the bounding box with semi-transparent fill
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                # Draw the bounding box border
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)

                # Add the header label
                label = f"run {run_id}"
                cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Write the frame to the video
            video_writer.write(frame)

            frame_id += 1

        # Release the video capture and writer
        cap.release()
        video_writer.release()

def make_static_graph():
    pass

def make_map():
    pass

def visualize_inference_results(detections, run_path):
    """
    Visualize and save videos with overlayed detection results.

    Args:
        detections (pandas.DataFrame): DataFrame with columns: [video, frame_id, run_id, class, x0, y0, x1, y1, confidence].
        run_path (str): Directory used to save visualizations.
    """
    make_video(detections, run_path)
