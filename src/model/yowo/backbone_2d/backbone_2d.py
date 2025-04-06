import torch.nn as nn
from .yolo_free import build_yolo_free


class Backbone2D(nn.Module):
    def __init__(self, backbone_2d, pretrained=False):
        super().__init__()

        if backbone_2d in ['yolo_free_pico', 'yolo_free_nano', 'yolo_free_tiny', 'yolo_free_large', 'yolo_free_huge']:
            print(f'Using {backbone_2d} as the 2D backbone.')
            self.backbone, self.feat_dims = build_yolo_free(backbone_2d, pretrained)

        else:
            print('Unknown 2D Backbone ...')
            exit()

    def forward(self, x):
        """
            Input:
                x: (Tensor) -> [B, C, H, W]
            Output:
            Output:
                y: (List) -> [
                    (Tensor) -> [B, C1, H1, W1],
                    (Tensor) -> [B, C2, H2, W2],
                    (Tensor) -> [B, C3, H3, W3]
                    ]
        """
        feat = self.backbone(x)

        return feat


def build_backbone_2d(backbone_2d, pretrained=False):
    backbone = Backbone2D(backbone_2d, pretrained)
    return backbone, backbone.feat_dims
