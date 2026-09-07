"""Baseline models: ResNet50, EfficientNet-B0, and ViT-Base, all via timm."""
import timm
import torch.nn as nn


def build_baseline(name: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    name_to_timm = {
        "resnet50": "resnet50",
        "efficientnet_b0": "efficientnet_b0",
        "vit_base": "vit_base_patch16_224",
    }
    if name not in name_to_timm:
        raise ValueError(f"Unknown baseline '{name}'. Choose from {list(name_to_timm)}.")
    return timm.create_model(name_to_timm[name], pretrained=pretrained, num_classes=num_classes)
