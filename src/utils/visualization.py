import os
import cv2
import numpy as np
import random
from pathlib import Path

import plotly.graph_objects as go


def make_video(detections, save_folder):
    grouped = detections.groupby('video')

    for video_name, video_detections in grouped:
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

                label = f"id {run_id}, c. {round(detection['confidence'], 2)}"
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


def visualize_inference_results(detections, save_folder):
    """
    Visualize and save videos with overlayed detection results.

    Args:
        detections (pandas.DataFrame): DataFrame with columns: [video, frame_id, run_id, class, x0, y0, x1, y1, confidence].
        save_folder (str): Directory used to save visualizations.
    """
    os.makedirs(save_folder, exist_ok=True)
    make_video(detections, save_folder)


def visualize_evaluation_results(angles: list,
                           durations: list,
                           path_graphs_gtvsp_angles: Path,
                           path_graphs_gtvsp_durations: Path,
                           path_graphs_duration_error_vs_duration: Path,
                                 angle_r2,
                                 duration_r2
                                 ):
    """
    Makes evaluation graphs using Plotly.

    Args:
        angles (List): List of two lists, [ground_truth_angles, measured_angles].
        durations (List): List of two lists, [ground_truth_durations, measured_durations].
        path_graphs_gtvsp_angles (Path): Path to save a graph with y=ground truth angles and x=measured angles.
        path_graphs_gtvsp_durations (Path): Path to save a graph with y=ground truth durations and x=measured durations.
        path_graphs_duration_error_vs_duration (Path): Path to save a graph with duration errors (diff gt-measured) as y and gt durations as x.
    """
    gt_angles, measured_angles = angles
    gt_durations, measured_durations = durations

    # Graph 1: Ground truth vs measured Angles
    fig_angles = go.Figure()
    fig_angles.add_trace(go.Scatter(
        x=measured_angles,
        y=gt_angles,
        mode='markers',
        marker=dict(color='#0e004f'),
        name=f'Angles (R²={angle_r2:.3f})'
    ))
    fig_angles.add_trace(go.Scatter(
        x=gt_angles,
        y=gt_angles,
        mode='lines',
        line=dict(color='#9b050c'),
        name='Identity line'
    ))
    fig_angles.update_layout(
        title='',
        xaxis_title='Predicted angles',
        yaxis_title='Ground truth angles',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
    )
    fig_angles.write_image(str(path_graphs_gtvsp_angles))

    # Graph 2: Ground Truth vs Measured Durations
    fig_durations = go.Figure()
    fig_durations.add_trace(go.Scatter(
        x=measured_durations,
        y=gt_durations,
        mode='markers',
        marker=dict(color='#0e004f'),
        name=f'Durations (R²={duration_r2:.3f})'
    ))
    fig_durations.add_trace(go.Scatter(
        x=gt_durations,
        y=gt_durations,
        mode='lines',
        line=dict(color='#9b050c'),
        name='Identity line'
    ))
    fig_durations.update_layout(
        title='',
        xaxis_title='Predicted durations',
        yaxis_title='Ground truth durations',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
    )
    fig_durations.write_image(str(path_graphs_gtvsp_durations))

    # Graph 3: Duration Error vs Ground Truth Durations
    duration_errors = [gt - m for gt, m in zip(gt_durations, measured_durations)]
    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(
        x=gt_durations,
        y=duration_errors,
        mode='markers',
        marker=dict(color='#0e004f'),
        name='Duration errors'
    ))
    fig_error.update_layout(
        title='',
        xaxis_title='Ground truth durations',
        yaxis_title='Duration errors',
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            linecolor='black',
            linewidth=1,
            mirror=True
        ),
    )
    fig_error.write_image(str(path_graphs_duration_error_vs_duration))
