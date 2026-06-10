"""EasyOCR engine wrapper — returns text + bounding boxes.

Engine is a pluggable interface; swap the backend in get_reader() without
touching callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

_reader: Optional[object] = None


def get_reader(languages: list[str] | None = None) -> object:
    global _reader
    if _reader is None:
        import easyocr  # deferred so import costs are only paid on first call

        _reader = easyocr.Reader(languages or ["en"], gpu=False, verbose=False)
    return _reader


def read_image(
    image: np.ndarray,
    languages: list[str] | None = None,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Run OCR on a preprocessed image array.

    Returns list of dicts: {text, bbox, confidence}.
    bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (top-left clockwise).
    """
    reader = get_reader(languages)
    raw = reader.readtext(image)
    results = []
    for bbox, text, conf in raw:
        if float(conf) >= min_confidence:
            results.append({"text": str(text).strip(), "bbox": bbox, "confidence": float(conf)})
    return results


def read_file(
    path: str | Path,
    languages: list[str] | None = None,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Convenience: load file → preprocess → OCR."""
    from ocr.preprocess import load_image, preprocess

    img = load_image(str(path))
    proc = preprocess(img)
    return read_image(proc, languages=languages, min_confidence=min_confidence)


def extract_text_lines(ocr_results: list[dict]) -> list[str]:
    """Flatten OCR results to a list of text strings (one per box)."""
    return [r["text"] for r in ocr_results if r["text"]]
