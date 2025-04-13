import os
import cv2
import numpy as np
import random
from pathlib import Path


def make_video(detections, save_folder):
    grouped = detections.groupby('video')

    for video_name, video_detections in grouped:
        os.makedirs(save_folder, exist_ok=True)
        output_video_path = (Path(save_folder) / Path(video_name).name.split('_')[0]).with_suffix(Path(video_name).suffix)
        input_video_path = os.path.join(video_name)
        cap = cv2.VideoCapture(input_video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_id = 0
        run_colors = {}
        vector_length = 50

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_detections = video_detections[video_detections['frame_id'] == frame_id]

            for _, detection in frame_detections.iterrows():
                x0 = int(detection['x0'])
                y0 = int(detection['y0'])
                x1 = int(detection['x1'])
                y1 = int(detection['y1'])
                run_id = detection['run_id']
                angle = detection['angle']

                if run_id not in run_colors:
                    hsv_color = cv2.applyColorMap(np.uint8([[random.randint(0, 255)]]), cv2.COLORMAP_HSV)[0][0]
                    run_colors[run_id] = (int(hsv_color[0]), int(hsv_color[1]), int(hsv_color[2]))
                color = run_colors[run_id]

                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)

                label = f"conf {round(detection['confidence'], 2)}"
                banner_width = x0 + len(label) * 10
                cv2.rectangle(frame, (x0, y0 - 30), (banner_width, y0), color, -1)
                cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                center_x = (x0 + x1) // 2
                center_y = (y0 + y1) // 2

                end_x = int(center_x + vector_length * np.sin(angle))
                end_y = int(center_y - vector_length * np.cos(angle))

                cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), color, 2, tipLength=0.3)

            video_writer.write(frame)
            frame_id += 1

        cap.release()
        video_writer.release()



def make_static_graph():
    pass


def visualize_inference_results(detections, save_folder):
    """
    Visualize and save videos with overlayed detection results.

    Args:
        detections (pandas.DataFrame): DataFrame with columns: [video, frame_id, run_id, class, x0, y0, x1, y1, confidence].
        save_folder (str): Directory used to save visualizations.
    """
    make_video(detections, save_folder)
