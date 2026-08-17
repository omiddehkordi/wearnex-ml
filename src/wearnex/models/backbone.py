"""Shared CNN backbone used by both the classifier and the embedding extractor.

Both downstream models are transfer-learned from the same pretrained
trunk so that, later, a single forward pass through the backbone can
feed both heads at inference time instead of running two full networks.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

_SUPPORTED = {"resnet50"}


def build_backbone(name: str = "resnet50", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return (feature-extractor module, output feature dimension).

    The returned module maps an (N, 3, H, W) image batch to an
    (N, feature_dim) feature vector — the classification/pooling layer
    is stripped off so callers attach their own heads.
    """
    if name not in _SUPPORTED:
        raise ValueError(f"Unsupported backbone {name!r}; choose from {_SUPPORTED}")

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.resnet50(weights=weights)
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, feature_dim


def freeze(module: nn.Module) -> None:
    """Freeze all parameters, e.g. for a warm-up phase that only trains new heads."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze(module: nn.Module) -> None:
    """Unfreeze all parameters, e.g. for a later fine-tuning phase."""
    for param in module.parameters():
        param.requires_grad = True
