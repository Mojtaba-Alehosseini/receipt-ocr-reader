"""FastAPI endpoint tests using the TestClient (no actual OCR invoked)."""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from ocr.schema import Receipt

client = TestClient(app)


def _make_png_bytes(h: int = 50, w: int = 50) -> bytes:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_extract_returns_receipt_json():
    mock_result = Receipt(merchant="TEST SHOP", date="2024-01-15", total=9.99)
    with (
        patch("api.main.read_image", return_value=[]),
        patch("api.main.extract_text_lines", return_value=[]),
        patch("api.main.extract", return_value=mock_result),
        patch("api.main.preprocess", return_value=np.zeros((50, 50), dtype=np.uint8)),
    ):
        resp = client.post(
            "/extract",
            files={"file": ("receipt.png", _make_png_bytes(), "image/png")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["merchant"] == "TEST SHOP"
    assert data["total"] == pytest.approx(9.99)


def test_extract_rejects_non_image():
    resp = client.post(
        "/extract",
        files={"file": ("doc.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 422


def test_extract_invalid_image_bytes():
    with patch("api.main.preprocess", return_value=np.zeros((50, 50), dtype=np.uint8)):
        resp = client.post(
            "/extract",
            files={"file": ("bad.jpg", b"not_an_image", "image/jpeg")},
        )
    assert resp.status_code == 422
