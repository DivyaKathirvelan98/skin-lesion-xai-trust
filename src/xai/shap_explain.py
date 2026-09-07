"""Model-agnostic SHAP explanations via GradientExplainer."""
import numpy as np
import shap
import torch


def shap_saliency(model: torch.nn.Module, background: torch.Tensor, image: torch.Tensor,
                   target_class: int) -> np.ndarray:
    """Return a normalized (H, W) SHAP attribution map for a single image.

    Args:
        model: classifier mapping (B, C, H, W) -> (B, num_classes) logits.
        background: a small batch of representative training images for the explainer baseline.
        image: single preprocessed tensor of shape (1, C, H, W).
        target_class: class index to explain.
    """
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(image, nsamples=50)
    # shap_values: list (per class) of arrays shaped like `image`
    class_attr = np.asarray(shap_values[target_class])[0]  # (C, H, W)
    saliency = np.abs(class_attr).sum(axis=0)
    saliency -= saliency.min()
    max_val = saliency.max()
    return saliency / max_val if max_val > 0 else saliency
