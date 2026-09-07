"""DullRazor-style hair/artifact removal for dermoscopic images."""
import cv2
import numpy as np


def remove_hair(image_bgr: np.ndarray, kernel_size: int = 17, inpaint_radius: int = 1) -> np.ndarray:
    """Suppress dark hair strands via black-hat morphology + inpainting.

    Args:
        image_bgr: input image, BGR, uint8, HxWx3.
        kernel_size: structuring element size for the black-hat filter.
        inpaint_radius: radius passed to cv2.inpaint.

    Returns:
        Hair-suppressed image, same shape/dtype as input.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image_bgr, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
