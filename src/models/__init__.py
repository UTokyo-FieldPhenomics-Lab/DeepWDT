from .yowo.build import build_yowo


def build_model(parameters, model_architecture, trainable=False):
    # build action detector
    if 'yowo_v2_' in parameters['MODEL_VERSION']:
        model, criterion = build_yowo(parameters, model_architecture, trainable)

    return model, criterion

