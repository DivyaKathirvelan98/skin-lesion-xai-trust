"""Cross-method XAI agreement: how consistently different explainers highlight the same region."""
import numpy as np
from scipy.stats import spearmanr


def binarize(saliency: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (saliency >= threshold).astype(np.uint8)


def iou(map_a: np.ndarray, map_b: np.ndarray, threshold: float = 0.5) -> float:
    bin_a, bin_b = binarize(map_a, threshold), binarize(map_b, threshold)
    intersection = np.logical_and(bin_a, bin_b).sum()
    union = np.logical_or(bin_a, bin_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def spearman_agreement(map_a: np.ndarray, map_b: np.ndarray) -> float:
    rho, _ = spearmanr(map_a.flatten(), map_b.flatten())
    return float(rho) if not np.isnan(rho) else 0.0


def cross_method_agreement(saliency_maps: dict, threshold: float = 0.5) -> dict:
    """Pairwise IoU and Spearman agreement across an arbitrary number of saliency maps.

    Args:
        saliency_maps: {method_name: (H, W) array in [0, 1]}, all resized to the same shape.

    Returns:
        {"pairwise_iou": {...}, "pairwise_spearman": {...}, "mean_iou": float, "mean_spearman": float}
    """
    names = list(saliency_maps.keys())
    pairwise_iou, pairwise_spearman = {}, {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}_vs_{names[j]}"
            pairwise_iou[key] = iou(saliency_maps[names[i]], saliency_maps[names[j]], threshold)
            pairwise_spearman[key] = spearman_agreement(saliency_maps[names[i]], saliency_maps[names[j]])

    mean_iou = float(np.mean(list(pairwise_iou.values()))) if pairwise_iou else 0.0
    mean_spearman = float(np.mean(list(pairwise_spearman.values()))) if pairwise_spearman else 0.0
    return {
        "pairwise_iou": pairwise_iou,
        "pairwise_spearman": pairwise_spearman,
        "mean_iou": mean_iou,
        "mean_spearman": mean_spearman,
    }
