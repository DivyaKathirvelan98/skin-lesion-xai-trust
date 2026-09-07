"""Quantitative XAI faithfulness metrics anchored to ground-truth lesion segmentation masks."""
import numpy as np


def saliency_vs_mask_iou(saliency: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """IoU between a thresholded saliency map and the ground-truth lesion segmentation mask."""
    sal_bin = (saliency >= threshold).astype(np.uint8)
    mask_bin = (gt_mask > 0).astype(np.uint8)
    intersection = np.logical_and(sal_bin, mask_bin).sum()
    union = np.logical_or(sal_bin, mask_bin).sum()
    return float(intersection / union) if union > 0 else 0.0


def pointing_game_hit(saliency: np.ndarray, gt_mask: np.ndarray) -> bool:
    """True if the saliency map's peak pixel falls inside the ground-truth lesion mask."""
    peak_idx = np.unravel_index(np.argmax(saliency), saliency.shape)
    return bool(gt_mask[peak_idx] > 0)


def deletion_insertion_auc(model, image, saliency: np.ndarray,
                            target_class: int, steps: int = 50, mode: str = "deletion") -> float:
    """Deletion/Insertion faithfulness metric (Petsiuk et al., 2018).

    Deletion: progressively zero out the most-salient pixels and track how fast the target
    class probability drops (a *faithful* explanation drops it fast -> low AUC is better).
    Insertion: the reverse -- start from a blurred/zero image and progressively reveal the
    most-salient pixels, tracking how fast probability rises (higher AUC is better).

    `model` is a torch.nn.Module and `image` a torch.Tensor; torch is imported lazily here so
    that the pure-numpy faithfulness metrics above stay usable without a torch installation.
    """
    import torch
    import torch.nn.functional as F

    h, w = saliency.shape
    order = np.argsort(-saliency.flatten())  # descending salience
    n_pixels = h * w
    step_size = max(n_pixels // steps, 1)

    if mode == "deletion":
        canvas = image.clone()
        baseline = torch.zeros_like(image)
    elif mode == "insertion":
        canvas = torch.zeros_like(image)
        baseline = image.clone()
    else:
        raise ValueError("mode must be 'deletion' or 'insertion'")

    scores = []
    flat_idx = 0
    with torch.no_grad():
        for _ in range(steps + 1):
            probs = F.softmax(model(canvas), dim=-1)[0, target_class].item()
            scores.append(probs)

            pixels_to_flip = order[flat_idx: flat_idx + step_size]
            rows, cols = np.unravel_index(pixels_to_flip, (h, w))
            canvas[..., rows, cols] = baseline[..., rows, cols]
            flat_idx += step_size

    return float(np.trapz(scores, dx=1.0 / len(scores)))
