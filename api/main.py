"""FastAPI service: POST /extract, GET /health."""

from __future__ import annotations

import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from ocr.engine import extract_text_lines, read_image
from ocr.extract import extract
from ocr.preprocess import preprocess
from ocr.schema import Receipt

app = FastAPI(title="Receipt OCR Reader", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract", response_model=Receipt)
async def extract_receipt(file: UploadFile = File(...)) -> Receipt:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="File must be an image")

    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}") from exc

    img = np.array(pil_img)
    # Convert RGB → BGR for OpenCV preprocessing
    img_bgr = img[:, :, ::-1].copy()
    proc = preprocess(img_bgr)

    ocr_results = read_image(proc)
    lines = extract_text_lines(ocr_results)
    return extract(lines)
