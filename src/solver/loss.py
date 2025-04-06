import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import SimOTA
from src.utils.box_ops import get_ious
from src.utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


class Criterion(object):
    def __init__(self, configuration):
        self.num_classes = 1
        self.img_size = configuration.dataset.image_size[0]
        self.loss_conf_weight = configuration.solver.loss.confidence_weight
        self.loss_cls_weight = configuration.solver.loss.class_weight
        self.loss_reg_weight = configuration.solver.loss.regression_weight
        self.focal_loss = configuration.solver.loss.focal_loss

        # Define loss functions
        self.obj_lossf = nn.BCEWithLogitsLoss(reduction='none')
        self.cls_lossf = nn.BCEWithLogitsLoss(reduction='none')

        # Instantiate matcher
        self.matcher = SimOTA(
            num_classes=self.num_classes,
            center_sampling_radius=configuration.model.center_sampling_radius,
            topk_candidate=configuration.model.top_k
        )

    def __call__(self, outputs, targets):
        """
            outputs['pred_conf']: List(Tensor) [B, M, 1]
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            targets: (List) [dict{'boxes': [...],
                                 'labels': [...],
                                 'orig_size': ...}, ...]
        """
        # Get output configuration
        batch_size = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']

        # Group predictions from all scales
        conf_preds = torch.cat(outputs['pred_conf'], dim=1)
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)

        cls_targets = []
        box_targets = []
        conf_targets = []
        fg_masks = []

        for batch_idx in range(batch_size):
            # Get target boxes and classes
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # Denormalize tgt_bbox
            tgt_bboxes *= self.img_size

            # Check if there is a target to find for the input
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                num_anchors = sum([ab.shape[0] for ab in anchors])
                cls_target = conf_preds.new_zeros((0, self.num_classes))
                box_target = conf_preds.new_zeros((0, 4))
                conf_target = conf_preds.new_zeros((num_anchors, 1))
                fg_mask = conf_preds.new_zeros(num_anchors).bool()
            else:
                # Get the best predictions for the targets
                (gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg_img,) = self.matcher(
                    fpn_strides=fpn_strides,
                    anchors=anchors,
                    pred_conf=conf_preds[batch_idx],
                    pred_cls=cls_preds[batch_idx],
                    pred_box=box_preds[batch_idx],
                    tgt_labels=tgt_labels,
                    tgt_bboxes=tgt_bboxes,
                )

                conf_target = fg_mask.unsqueeze(-1)
                box_target = tgt_bboxes[matched_gt_inds]
                cls_target = F.one_hot(gt_matched_classes.long(), self.num_classes)
                cls_target = cls_target * pred_ious_this_matching.unsqueeze(-1)

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            conf_targets.append(conf_target)
            fg_masks.append(fg_mask)

        # Group matched
        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        conf_targets = torch.cat(conf_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_foregrounds = fg_masks.sum()

        # Compute confidence loss
        loss_conf = self.obj_lossf(conf_preds.view(-1, 1), conf_targets.float())
        loss_conf = loss_conf.sum() / num_foregrounds

        # Compute class loss
        matched_cls_preds = cls_preds.view(-1, self.num_classes)[fg_masks] # Get classes predictions for foreground objects
        loss_cls = self.cls_lossf(matched_cls_preds, cls_targets)
        loss_cls = loss_cls.sum() / num_foregrounds

        # Compute box loss
        matched_box_preds = box_preds.view(-1, 4)[fg_masks] # Get boxes predictions for foreground objects
        ious = get_ious(matched_box_preds,
                        box_targets,
                        box_mode="xyxy",
                        iou_type='giou')
        loss_box = (1.0 - ious).sum() / num_foregrounds

        # Compute total loss
        losses = self.loss_conf_weight * loss_conf + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_reg_weight * loss_box

        loss_dict = dict(
            loss_conf=loss_conf,
            loss_cls=loss_cls,
            loss_box=loss_box,
            losses=losses
        )

        return loss_dict


def build_loss(configuration):
    criterion = Criterion(configuration)

    return criterion
