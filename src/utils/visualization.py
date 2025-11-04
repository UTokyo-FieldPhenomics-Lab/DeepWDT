import os
import cv2
import numpy as np
import random
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go


def make_inference_video(detections, save_folder, duration_measurement_method = 'range'):
    grouped = detections.groupby('video')

    for video_name, video_detections in grouped:
        output_video_path = (Path(save_folder) / Path(video_name).name)
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

                run_df = video_detections[video_detections['run_id'] == detection['run_id']]['frame_id']
                if duration_measurement_method == 'count':
                    run_duration = len(run_df)

                elif duration_measurement_method == 'range':
                    run_duration = run_df.max() - run_df.min() + 1

                label = f"id {run_id}, c. {round(detection['confidence'], 2)}, dur. {run_duration}, ang. {round(detection['angle'], 2)}"
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


def make_evaluation_graphs(angles: list,
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
        x=gt_angles,
        y=measured_angles,
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
        yaxis_title='Predicted angles',
        xaxis_title='Ground truth angles',
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
        x=gt_durations,
        y=measured_durations,
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
        yaxis_title='Predicted durations',
        xaxis_title='Ground truth durations',
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


def make_evaluation_trajectory_graphs(ground_truth, detections, matches, trajectory_graphs_path, graph_size):
    """
    Creates compared run visualization graphs.
    For each unique video / run_id group in matches, it should make a png with two graphs, one for ground truth and one for detected runs. These graphs are horizontally concatenated.
    Each has one point for each bounding box center (one bbox for one frame detection). Colors of the points follow magma (gt) and cividis (detection) charts in opposite orders.
    Graphs are each 448x448 pixels, and the grid is at every 100 pixels, light gray. Both graphs are next to each other.
    Then a dark red line is added to show the angle. Angles in the dataframes are oriented so that 0 is upward and clockwise positive / anticlockwise negative, and in radians.

    Args:
        ground_truth (DataFrame): Ground truth runs dataframe. Has 8 columns: "video", "run_id", "frame_id", "x0", "x1", "y0", "y1", "angle". Coordinates are absolute.
        detections (DataFrame): Detected runs dataframe. Has 8 columns: "video", "run_id" "frame_id", "x0", "x1", "y0", "y1", "angle". Coordinates are absolute.
        matches (DataFrame): Matched id between ground truth runs and detected runs. Has 3 columns: "video" (video name), "gt_run_id", "detection_run_id".
        trajectory_graphs_path (Path): Path to save trajectory graphs.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import os

    # Create directory if it doesn't exist
    os.makedirs(trajectory_graphs_path, exist_ok=True)

    # Set figure size constants
    GRAPH_SIZE = graph_size
    GRID_STEP = 100

    # Process each matching pair of runs
    for _, match in matches.iterrows():
        video = match['video']
        gt_run_id = match['gt_run_id']
        detection_run_id = match['detection_run_id']

        # Filter dataframes to get the specific runs
        gt_run = ground_truth[(ground_truth['video'] == video) & 
                              (ground_truth['run_id'] == gt_run_id)].copy()

        detected_run = detections[(detections['video'] == video) & 
                                  (detections['run_id'] == detection_run_id)].copy()

        if gt_run.empty or detected_run.empty:
            continue

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(GRAPH_SIZE*2/100, GRAPH_SIZE/100), dpi=100)

        # Calculate center points for each bounding box
        gt_run['center_x'] = (gt_run['x0'] + gt_run['x1']) / 2
        gt_run['center_y'] = (gt_run['y0'] + gt_run['y1']) / 2
        detected_run['center_x'] = (detected_run['x0'] + detected_run['x1']) / 2
        detected_run['center_y'] = (detected_run['y0'] + detected_run['y1']) / 2

        # Setup colormaps - magma for ground truth (in reverse) and cividis for detections (in reverse)
        gt_cmap = cm.get_cmap('magma_r')
        det_cmap = cm.get_cmap('cividis_r')

        # Normalize frame IDs for coloring
        if len(gt_run) > 1:
            gt_min_frame = gt_run['frame_id'].min()
            gt_max_frame = gt_run['frame_id'].max()
            gt_norm = plt.Normalize(gt_min_frame, gt_max_frame)
        else:
            gt_norm = plt.Normalize(0, 1)

        if len(detected_run) > 1:
            det_min_frame = detected_run['frame_id'].min()
            det_max_frame = detected_run['frame_id'].max()
            det_norm = plt.Normalize(det_min_frame, det_max_frame)
        else:
            det_norm = plt.Normalize(0, 1)

        # Plot ground truth trajectory (left plot)
        for i, row in gt_run.iterrows():
            color = gt_cmap(gt_norm(row['frame_id']))
            ax1.scatter(row['center_x'], row['center_y'], color=color, s=10)

        # Plot detected trajectory (right plot)
        for i, row in detected_run.iterrows():
            color = det_cmap(det_norm(row['frame_id']))
            ax2.scatter(row['center_x'], row['center_y'], color=color, s=10)

        # Add angle vectors using the first detection's angle for each run
        if not gt_run.empty:
            first_gt = gt_run.iloc[0]
            center_x, center_y = first_gt['center_x'], first_gt['center_y']
            angle = first_gt['angle']
            # Convert angle to vector (0 is upward, clockwise positive)
            vector_length = 50  # Length of the vector line
            dx = vector_length * np.sin(angle)
            dy = -vector_length * np.cos(angle)  # Negative because y-axis is inverted in image coords
            ax1.arrow(center_x, center_y, dx, dy, color='darkred', width=1, head_width=5, length_includes_head=True)

        if not detected_run.empty:
            first_det = detected_run.iloc[0]
            center_x, center_y = first_det['center_x'], first_det['center_y']
            angle = first_det['angle']
            vector_length = 50
            dx = vector_length * np.sin(angle)
            dy = -vector_length * np.cos(angle)  # Negative because y-axis is inverted in image coords
            ax2.arrow(center_x, center_y, dx, dy, color='darkred', width=1, head_width=5, length_includes_head=True)

        # Configure both plots with grid lines and equal aspects
        for ax in [ax1, ax2]:
            # Set grid lines at every GRID_STEP pixels
            ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
            ax.set_xticks(np.arange(0, GRAPH_SIZE+1, GRID_STEP))
            ax.set_yticks(np.arange(0, GRAPH_SIZE+1, GRID_STEP))

            # Set aspect ratio to be equal
            ax.set_aspect('equal')

            # Set limits to match the image size
            ax.set_xlim(0, GRAPH_SIZE)
            ax.set_ylim(0, GRAPH_SIZE)

        # Set titles
        ax1.set_title(f'Ground truth')
        ax2.set_title(f'Prediction')

        # Save the figure
        plt.tight_layout()
        output_file = trajectory_graphs_path / f"{video}_gt{gt_run_id}_det{detection_run_id}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
