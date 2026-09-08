"""Baseline models: ResNet50, EfficientNet-B0, and ViT-Base, all via timm."""
import timm
import torch.nn as nn


def build_baseline(name: str, num_classes: int = 7, pretrained: bool = True, image_size: int = 224) -> nn.Module:
    name_to_timm = {
        "resnet50": "resnet50",
        "efficientnet_b0": "efficientnet_b0",
        "vit_base": "vit_base_patch16_224",
    }
    if name not in name_to_timm:
        raise ValueError(f"Unknown baseline '{name}'. Choose from {list(name_to_timm)}.")

    if name == "vit_base" and image_size != 224:
        # ViT's patch embedding + positional embedding are baked in for 224x224 by name;
        # dynamic_img_size lets it accept other resolutions via interpolated pos embeddings.
        return timm.create_model(
            name_to_timm[name], pretrained=pretrained, num_classes=num_classes,
            img_size=image_size, dynamic_img_size=True,
        )
    return timm.create_model(name_to_timm[name], pretrained=pretrained, num_classes=num_classes)
