import torch
from .yowo import YOWO
from .loss import build_criterion


# build YOWO detector
def build_yowo(parameters, model_architecture, trainable=False):

    # build YOWO
    model = YOWO(parameters, model_architecture, trainable = trainable,)

    if trainable:
        # Freeze backbone
        if parameters['FREEZE_BACKBONE_2D']:
            for m in model.backbone_2d.parameters():
                m.requires_grad = False
        if parameters['FREEZE_BACKBONE_3D']:
            for m in model.backbone_3d.parameters():
                m.requires_grad = False
            
        # keep training
        if parameters['RESUME']:
            checkpoint = torch.load(resume, map_location='cpu')
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)

        # build criterion
        criterion = build_criterion(
            parameters,
            parameters['IMAGE_SIZE'],
            parameters['CLASSES'],
            parameters['MULTI_HOT'],
        )
    
    else:
        criterion = None
                        
    return model, criterion
