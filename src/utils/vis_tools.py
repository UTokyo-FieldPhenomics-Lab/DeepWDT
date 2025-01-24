import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd

def vis_targets(video_clips, targets):
    """
        video_clips: (Tensor) -> [B, C, T, H, W]
        targets: List[Dict] -> [{'boxes': (Tensor) [N, 4],
                                 'labels': (Tensor) [N,]}, 
                                 ...],
    """
    batch_size = len(video_clips)

    for batch_index in range(batch_size):
        video_clip = video_clips[batch_index]
        target = targets[batch_index]

        key_frame = video_clip[:, :, -1, :, :]
        tgt_bboxes = target['boxes']
        tgt_labels = target['labels']

        key_frame = convert_tensor_to_cv2img(key_frame)
        width, height = key_frame.shape[:-1]

        for box, label in zip(tgt_bboxes, tgt_labels):
            x1, y1, x2, y2 = box
            label = int(label)

            x1 *= width
            y1 *= height
            x2 *= width
            y2 *= height

            # draw bbox
            cv2.rectangle(key_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (255, 0, 0), 2)
        cv2.imshow('groundtruth', key_frame)
        cv2.waitKey(0)


def convert_tensor_to_cv2img(img_tensor):
    """ convert torch.Tensor to cv2 image """
    # to numpy
    img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()
    # to cv2 img Mat
    cv2_img = img_tensor.astype(np.uint8)
    # to BGR
    cv2_img = cv2_img.copy()[..., (2, 1, 0)]

    return cv2_img


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def vis_detection(frame, scores, labels, bboxes, vis_thresh, class_names, class_colors):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            label = int(labels[i])
            cls_color = class_colors[label]
                
            if len(class_names) > 1:
                mess = 'Dance'
            else:
                cls_color = [144, 0, 190]
                mess = None
                # visualize bbox
            frame = plot_bbox_labels(frame, bbox, mess, cls_color, text_scale=ts)

    return frame


def trajectory_visualization(angle, df_id, filename, trajectory_size, path_evaluation):
    """
    creates run visualization graphs
    """
    fig, ax = plt.subplots(figsize=(trajectory_size/100, trajectory_size/100))
    ax.set_yticks(np.arange(0, trajectory_size+1, 100))
    ax.axis([0, trajectory_size, 0, trajectory_size])
    ax.set_facecolor('white')
    ax.grid(True)
    
    centers = df_id[['center_x', 'center_y']].values
    centers[:, 1] = trajectory_size - centers[:, 1]

    colors = plt.cm.cividis(np.linspace(0, 1, len(centers)))[::-1]
    ax.scatter(centers[:, 0], centers[:, 1], c=colors, s=5)

    length = math.sqrt((centers[0][0]-centers[-1][0])**2+(centers[0][1]-centers[-1][1])**2)/2+30
    mean_center = np.mean(centers, axis=0)

    dx1, dy1 = length * np.cos(angle), length * np.sin(angle + np.pi)
    dx2, dy2 = length * np.cos(angle + np.pi), length * np.sin(angle)

    end_point1 = (mean_center[0] + dx1, mean_center[1] + dy1)
    end_point2 = (mean_center[0] + dx2, mean_center[1] + dy2)

    ax.plot([end_point1[0], end_point2[0]], [end_point1[1], end_point2[1]], color='darkred')

    output_dir = os.path.join(path_evaluation, 'trajectories')
    fig.savefig(os.path.join(output_dir, filename), dpi=100)
    plt.close(fig)


def make_videos(dataset, video_format):
    path_inference = 'runs/inference/'
    path_tubes = os.path.join(path_inference, dataset, 'tubes')
    path_videos = os.path.join('data', dataset, 'videos')
    path_save = os.path.join(path_inference, dataset, 'videos')
    os.makedirs(path_save, exist_ok=True)

    for video_file in os.listdir(path_tubes):
        tubes = pd.read_csv(os.path.join(path_tubes, video_file))

        video_name = os.path.splitext(video_file)[0] + f'.{video_format}'
        video_path = os.path.join(path_videos, video_name)

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(path_save, video_name), fourcc, fps, (width, height))

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_boxes = tubes[tubes['frame_id'] == frame_id]

            for _, box in current_boxes.iterrows():
                x1, y1, x2, y2 = int(box['x0']), int(box['y0']), int(box['x1']), int(box['y1'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), [144, 0, 190], 2)
                # cv2.putText(frame, f"Dance ID: {box['run_id']}", (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame)
            frame_id += 1

        cap.release()
        out.release()
