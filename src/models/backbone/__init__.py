from .backbone_2d.backbone_2d import Backbone2D
from .backbone_3d.backbone_3d import Backbone3D


def build_backbone_2d(backbone_2d, pretrained=False):
    backbone = Backbone2D(backbone_2d, pretrained)
    return backbone, backbone.feat_dims


def build_backbone_3d(backbone_3d, model_size, pretrained=False):
    backbone = Backbone3D(backbone_3d, model_size, pretrained)
    return backbone, backbone.feat_dim

