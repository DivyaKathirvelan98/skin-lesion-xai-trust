"""Grad-CAM++ explanations for the CNN stem, via the `grad-cam` package."""
import numpy as np
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def gradcam_saliency(model: torch.nn.Module, target_layer, image: torch.Tensor, target_class: int) -> np.ndarray:
    """Return a normalized (H, W) saliency map in [0, 1] for a single image.

    `image` is a single preprocessed tensor of shape (1, C, H, W).
    `target_layer` is the last convolutional layer of the CNN stem to hook into.
    """
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=image, targets=targets)[0]
    return grayscale_cam  # already normalized to [0, 1] by the library
