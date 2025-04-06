import numpy as np


def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep = np.zeros(len(bboxes), dtype=np.int32)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)


def compute_iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    Each box is defined as [x0, y0, x1, y1].
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def grouped_nms(detections, threshold=0.5):
    """
    Remove non-necessary bounding boxes using non-maximum suppression (NMS).

    Args:
        detections (pandas.DataFrame): DataFrame containing detections with columns:
            [video, frame, class, x0, y0, x1, y1, confidence].
        threshold (float): IoU threshold to use for suppression.

    Returns:
        pandas.DataFrame: DataFrame with suppressed detections removed.
    """
    # List to hold the indices of detections we want to keep.
    keep_indices = []

    # Group the detections by video, frame, and class to apply NMS within each group.
    for group_keys, group in detections.groupby(['video', 'frame_id', 'class']):
        group = group.sort_values(by='confidence', ascending=False)

        boxes = group[['x0', 'y0', 'x1', 'y1']].values
        indices = group.index.tolist()

        suppressed = set()

        # Iterate over the detections in each group
        for i in range(len(boxes)):
            if indices[i] in suppressed:
                continue
            # Keep the current box
            keep_indices.append(indices[i])
            # Compare this box with all the following boxes
            for j in range(i + 1, len(boxes)):
                if indices[j] in suppressed:
                    continue
                # Compute the IoU between the current box and the j-th box
                iou = compute_iou(boxes[i], boxes[j])
                # If the IoU exceeds the threshold, suppress the box
                if iou > threshold:
                    suppressed.add(indices[j])

    # Return the detections corresponding to the kept indices
    return detections.loc[keep_indices]