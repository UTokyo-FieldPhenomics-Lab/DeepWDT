from pathlib import Path
from tqdm import tqdm

import cv2
import pandas as pd
import torch
from PIL import Image

from src.model import build_yowo_model, track
from src.utils import grouped_nms, thieve_confidence, visualize_inference_results
from src.dataset import EvalTransform, find_closer_32k
from src.config import load_configuration


def infer_function(path_configuration, run_path):
    infer_configuration = load_configuration(path_configuration).infer
    print("Arguments: ", infer_configuration)
    print('-----------------------------------------------------------------------------------------------------------')

    # Get video names
    video_folder = Path('data') / infer_configuration.dataset.name / 'videos'
    videos = [f for f in video_folder.iterdir() if f.suffix in ('.mp4', '.MOV')]

    # Instantiate the model
    print('Building the model...')
    device = infer_configuration.device
    model = build_yowo_model(model_configuration=infer_configuration.model,
                             device=device,
                             trainable=False,
                             centered_clip=infer_configuration.dataset.centered_clip)
    model.eval()
    model.trainable = False
    model = model.to(device).train()
    print('Model built!')

    for video_name in videos:
        detection_bboxes = []
        detection_frames = []
        detection_confidence = []
        detection_videonames = []
        detection_classes = []

        # Load the video
        cap = cv2.VideoCapture(str(video_name))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Processing video: {video_name} with {total_frames} frames")

        # Prepare inference configuration
        downscaled_image_size = find_closer_32k([width * infer_configuration.downscale_factor, height * infer_configuration.downscale_factor])
        transform = EvalTransform(img_size = downscaled_image_size)

        # Load all frames into a list
        frames = []
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)
            frames.append(pil_frame)

        cap.release()

        # Build sliding window clips over the frames
        for i in tqdm(range(infer_configuration.dataset.len_clip - 1, len(frames)), desc=f'Video {video_name}'):
            # Prepare a clip
            clip = frames[i - infer_configuration.dataset.len_clip + 1: i + 1]
            clip_transformed, _ = transform(clip, None)
            clip_tensor = torch.stack(clip_transformed, dim=1)
            clip_tensor = clip_tensor.unsqueeze(0).to(device)

            # Run inference on the clip
            outputs = model(clip_tensor)
            scores, labels, bboxes = outputs

            # Process detections for this clip
            detection_bboxes.extend(bboxes[0][detection] for detection in range(len(scores[0])))
            detection_frames.extend(i for k in range(len(scores[0])))
            detection_confidence.extend(scores[0][detection] for detection in range(len(scores[0])))
            detection_videonames.extend(str(video_name) for detection in range(len(scores[0])))
            detection_classes.extend(int(labels[0][detection]) for detection in range(len(scores[0])))

        df_results = pd.DataFrame({
            'video': detection_videonames,
            'frame_id': detection_frames,
            'class': detection_classes,
            'x0': [bbox[0] for bbox in detection_bboxes],
            'y0': [bbox[1] for bbox in detection_bboxes],
            'x1': [bbox[2] for bbox in detection_bboxes],
            'y1': [bbox[3] for bbox in detection_bboxes],
            'confidence': detection_confidence
        })

        # Ensure the right formats in the result dataframe
        df_results['frame_id'] = df_results['frame_id'].astype(int)
        df_results['class'] = df_results['class'].astype(int)

        # Apply thieving and nms operations
        df_results = thieve_confidence(df_results, infer_configuration.confidence_threshold)
        df_results = grouped_nms(df_results, infer_configuration.nms_threshold)

        # Track dancing bees to produce runs
        df_results = track(df_results,
                           iou_threshold=infer_configuration.track_iou_threshold,
                           duration_threshold=infer_configuration.track_duration_threshold)

        # Convert relative coordinates to absolute coordinates
        if len(df_results) > 0:
            df_results[['x0', 'x1',]] = df_results[['x0', 'x1',]] * width
            df_results[['y0', 'y1']] = df_results[['y0', 'y1']] * height

        # Save and visualize results
        visualize_inference_results(df_results, run_path)

        # Translate dances to coordinates

        # Map coordinates
