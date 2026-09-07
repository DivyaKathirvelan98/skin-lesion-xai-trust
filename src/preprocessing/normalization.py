"""Illumination correction and contrast normalization for dermoscopic images."""
import cv2
import numpy as np


def gray_world_correction(image_bgr: np.ndarray) -> np.ndarray:
    """Simple gray-world illumination normalization."""
    image = image_bgr.astype(np.float32)
    mean_per_channel = image.reshape(-1, 3).mean(axis=0)
    gray_mean = mean_per_channel.mean()
    scale = gray_mean / np.clip(mean_per_channel, 1e-6, None)
    corrected = image * scale
    return np.clip(corrected, 0, 255).astype(np.uint8)


def clahe_on_luminance(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Apply CLAHE to the L channel of the LAB representation."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def normalize_image(image_bgr: np.ndarray, use_clahe: bool = True, clip_limit: float = 2.0) -> np.ndarray:
    """Full normalization: gray-world correction followed by optional CLAHE."""
    corrected = gray_world_correction(image_bgr)
    if use_clahe:
        corrected = clahe_on_luminance(corrected, clip_limit=clip_limit)
    return corrected
