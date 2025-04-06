import pandas as pd
import torch
from tqdm import tqdm

from src.dataset import build_dataset, CollateFunction, EvalTransform
from src.evaluation import get_metrics
from src.model import build_yowo_model, track
from src.utils import thieve_confidence, grouped_nms
from src.config import load_configuration


def eval_function(path_configuration):
    configuration = load_configuration(path_configuration)
    eval_configuration = configuration.eval

    print("Arguments: ", configuration)
    print('-----------------------------------------------------------------------------------------------------------')

    # Instantiate the validation dataset and dataloader
    print('Loading the validation dataset...')
    validation_dataset = build_dataset(
        configuration=eval_configuration.dataset,
        transform=EvalTransform(img_size=eval_configuration.dataset.image_size[0]),
        split=eval_configuration.split
    )

    evaluation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=eval_configuration.batch_size,
        collate_fn=CollateFunction(),
        num_workers=eval_configuration.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    print('Validation dataset loaded!')

    # Instantiate the model
    print('Building the model...')
    device = eval_configuration.device
    model = build_yowo_model(model_configuration=eval_configuration.model, device=device, trainable=True)
    checkpoint_path = torch.load(eval_configuration.weights, map_location='cpu').pop("model")
    model.load_state_dict(checkpoint_path)
    model = model.to(device).eval()
    model.trainable = False
    print('Model built!')

    # Evaluate the model
    print('Evaluating the model...')
    model.eval()

    # Detect dancing bees
    print('1. Detecting dancing bees...')

    results_list = []
    for iter_i, (frame_ids, video_clips, targets) in enumerate(tqdm(evaluation_dataloader)):

        # Model inference
        video_clips = video_clips.to(device)
        outputs = model(video_clips)  # scores, labels, bboxes

        # Add results to a dataframe
        scores, labels, bboxes = outputs
        for index, (score, label, bboxe) in enumerate(zip(scores, labels, bboxes)):

            for detection_id in range(len(score)):
                results_list.append({
                    'video': targets[index]['video'],
                    'frame_id': int(targets[index]['image_id']),
                    'class': int(label[detection_id]),
                    'x0': bboxe[detection_id][0],
                    'y0': bboxe[detection_id][1],
                    'x1': bboxe[detection_id][2],
                    'y1': bboxe[detection_id][3],
                    'confidence': score[detection_id]
                })

    detections = pd.DataFrame(results_list)
    detections['frame_id'] = detections['frame_id'].astype(int)
    detections['class'] = detections['class'].astype(int)

    # Thieve results based on the confidence
    detections = thieve_confidence(detections, eval_configuration.confidence_threshold)

    # Apply NMS to results
    detections = grouped_nms(detections, eval_configuration.nms_threshold)

    # Track dancing bees through frames
    print('2. Tracking dancing bees...')
    detections = track(detections)

    # Compute evaluation metrics
    print('3. Computing evaluation metrics...')
    if len(detections) == 0:
        print(f'[Detected runs: 0%]')
    else:
        detections[['x0', 'x1', 'y0', 'y1']] = detections[['x0', 'x1', 'y0', 'y1']] * 224
        eval_metrics = get_metrics(detections, validation_dataset.df)
        print(f'[Detected runs: {eval_metrics["detected_runs"]}%][Angle RMSE: {eval_metrics["angle_rmse"]}][Duration RMSE: {eval_metrics["duration_rmse"]}]')
