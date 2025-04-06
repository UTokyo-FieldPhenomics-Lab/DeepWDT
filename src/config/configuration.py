from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import List, Literal, Union, Optional


@dataclass
class DatasetConfig:
    name: str
    image_size: List[int]
    len_clip: int
    classes: List[str] = field(default_factory=lambda: ['Dancing'])
    centered_clip: bool = False

@dataclass
class Backbone2DConfig:
    name: str
    frozen: bool = False
    pretrained: bool = True
    stride: list = field(default_factory=lambda: [8,16,32])

@dataclass
class Backbone3DConfig:
    name: str
    frozen: bool = False
    pretrained: bool = True
    memory_momentum: float = 0.9
    model_size: str = "1.0x"

@dataclass
class ModelConfig:
    backbone_2d: Backbone2DConfig
    backbone_3d: Backbone3DConfig
    resume: Optional[str] = None
    name: str = 'yowo'
    center_sampling_radius: float  = 2.5
    confidence_threshold: float  = 0.01
    nms_threshold: float  = 0.5
    top_k: int  = 10
    head_dim: int = 64
    head_norm: str = 'BN'
    head_act: str = 'lrelu'
    num_cls_heads: int = 2
    num_reg_heads: int = 2
    head_depthwise: bool = True

@dataclass
class AugmentationConfig:
    exposure: float = 1.5
    hue: float = 0.1
    jitter: float = 0.2
    saturation: float = 1.5

@dataclass
class LossConfig:
    confidence_weight: int = 1
    class_weight: int = 0
    regression_weight: int = 5
    focal_loss: bool = False

@dataclass
class SolverConfig:
    loss: LossConfig
    accumulate: int = 1
    base_lr: float =  0.00005
    min_lr_ratio: float = 0.25
    warmup_epoch: int = 1
    no_decrease_lr_epoch: int = 5
    optimizer: str = 'adaw'
    momentum: float = 0.9
    weight_decay: float = 0.0005
    wp_iter: int = 500

@dataclass
class TrainConfig:
    dataset: DatasetConfig
    model: ModelConfig
    augmentation: AugmentationConfig
    solver: SolverConfig
    device: str = 'cuda'
    max_epoch: int = 20
    mlflow: bool = True
    batch_size: int = 64
    num_workers: int = 8

@dataclass
class EvalConfig:
    device: str
    weights: str
    dataset: DatasetConfig
    model: ModelConfig
    tracking_method: str = 'sort'
    split: str = 'val'
    batch_size: int = 64
    num_workers: int = 8
    confidence_threshold: float = 0.01
    nms_threshold: float = 0.5
    iou_threshold: float = 0.5
    track_iou_threshold: float = 0.3
    track_duration_threshold: int = 15

@dataclass
class InferConfig:
    dataset: DatasetConfig
    model: ModelConfig
    batch_size: int = 64
    num_workers: int = 8
    device: str = 'cuda'
    downscale_factor: float = 0.5
    confidence_threshold: float = 0.01
    nms_threshold: float = 0.5
    iou_threshold: float = 0.5
    track_iou_threshold: float = 0.3
    track_duration_threshold: int = 15

@dataclass
class Config:
    train: Optional[TrainConfig] = None
    eval: Optional[EvalConfig] = None
    infer: Optional[InferConfig] = None


def load_configuration(path_configuration):
    yaml_config = OmegaConf.load(path_configuration)
    configuration = OmegaConf.merge(OmegaConf.structured(Config), yaml_config)
    return configuration