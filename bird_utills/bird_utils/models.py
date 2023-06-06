from torch import nn

import torch
import torchvision.models as models

NUM_BIRD_CLASSES = 555

def get_frozen_model(
    base_vgg_model: torch.nn.Module,
    num_classes: int = NUM_BIRD_CLASSES) -> torch.nn.Module:
    # Freeze feauture layers
    for param in base_vgg_model.features.parameters():
        param.requires_grad = False

    # Modify last layer to have correct number of outputs
    base_vgg_model.classifier[-1] = torch.nn.Linear(4096, num_classes)

    return base_vgg_model

def get_vgg_19(
    num_classes: int = NUM_BIRD_CLASSES
) -> torch.nn.Module:
    return get_frozen_model(
        models.vgg19(pretrained = True),
        num_classes = num_classes
    )

def get_vgg_16(
    num_classes: int = NUM_BIRD_CLASSES
) -> torch.nn.Module:
    return get_frozen_model(
        models.vgg16(pretrained = True),
        num_classes = num_classes
    )

def get_vgg_13(
    num_classes: int = NUM_BIRD_CLASSES
) -> torch.nn.Module:
    return get_frozen_model(
        models.vgg13(pretrained = True),
        num_classes = num_classes
    )

def get_vgg_11(
    num_classes: int = NUM_BIRD_CLASSES
) -> torch.nn.Module:
    return get_frozen_model(
        models.vgg19(pretrained = True),
        num_classes = num_classes
    )