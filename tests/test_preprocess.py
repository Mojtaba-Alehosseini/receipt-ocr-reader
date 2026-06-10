"""Preprocessing pipeline tests — no real images, just array shapes."""

import numpy as np

from ocr.preprocess import binarise, deskew, preprocess, to_gray


def _make_bgr(h: int = 100, w: int = 80) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_gray(h: int = 100, w: int = 80) -> np.ndarray:
    return np.random.randint(0, 255, (h, w), dtype=np.uint8)


def test_to_gray_from_bgr():
    img = _make_bgr()
    out = to_gray(img)
    assert out.ndim == 2
    assert out.shape == (100, 80)


def test_to_gray_passthrough():
    img = _make_gray()
    out = to_gray(img)
    assert out.shape == (100, 80)


def test_deskew_returns_same_shape():
    gray = _make_gray()
    out = deskew(gray)
    assert out.shape == gray.shape


def test_binarise_output_binary():
    gray = _make_gray()
    out = binarise(gray)
    assert out.shape == gray.shape
    unique = np.unique(out)
    assert set(unique).issubset({0, 255})


def test_preprocess_pipeline():
    img = _make_bgr()
    out = preprocess(img)
    assert out.ndim == 2
    assert out.shape == (100, 80)
