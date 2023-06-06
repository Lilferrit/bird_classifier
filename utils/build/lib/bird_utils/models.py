from torch import nn

import torch
import torchvision.models as models

NUM_BIRD_CLASSES = 555

def freeze_features(
    base_vgg_model: torch.nn.Module,
) -> torch.nn.Module:
    # Freeze feauture layers
    for param in base_vgg_model.features.parameters():
        param.requires_grad = False

    return base_vgg_model

def get_frozen_model(
    base_vgg_model: torch.nn.Module,
    num_classes: int = NUM_BIRD_CLASSES) -> torch.nn.Module:
    # Freeze feauture layers
    freeze_features(base_vgg_model)

    # Modify last layer to have correct number of outputs
    base_vgg_model.classifier[-1] = torch.nn.Linear(4096, num_classes)

    return base_vgg_model

def get_vgg_19(
    num_classes: int = NUM_BIRD_CLASSES
) -> torch.nn.Module:
    return get_frozen_model(
        models.vgg19(weights = models.vgg.VGG19_Weights.DEFAULT),
        num_classes = num_classes
    )

def get_vgg_16(
    num_classes: int = NUM_BIRD_CLASSES
) -> torch.nn.Module:
    return get_frozen_model(
        models.vgg16(weights = models.vgg.VGG16_Weights.DEFAULT),
        num_classes = num_classes
    )

def get_vgg_13(
    num_classes: int = NUM_BIRD_CLASSES
) -> torch.nn.Module:
    return get_frozen_model(
        models.vgg13(weights = models.vgg.VGG13_Weights.DEFAULT),
        num_classes = num_classes
    )

def get_vgg_11(
    num_classes: int = NUM_BIRD_CLASSES
) -> torch.nn.Module:
    return get_frozen_model(
        models.vgg11(weights = models.vgg.VGG11_Weights.DEFAULT),
        num_classes = num_classes
    )