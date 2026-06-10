"""OpenCV preprocessing: deskew, denoise, binarise."""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Read image from file as BGR uint8 array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate image to align text lines horizontally."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    # Only correct small skews (< 10 degrees); larger could be rotated content
    if abs(median_angle) > 10:
        return gray

    h, w = gray.shape[:2]
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def binarise(gray: np.ndarray) -> np.ndarray:
    """Adaptive threshold; falls back to Otsu when image is mostly clean."""
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def preprocess(image: np.ndarray) -> np.ndarray:
    """Full pipeline: gray → deskew → binarise.  Returns uint8 array."""
    gray = to_gray(image)
    gray = deskew(gray)
    return binarise(gray)
