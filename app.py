"""Gradio demo: upload a receipt → OCR overlay + extracted fields + CSV export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from ocr.engine import extract_text_lines, read_image
from ocr.extract import extract
from ocr.preprocess import preprocess


def _draw_boxes(image: np.ndarray, ocr_results: list[dict]) -> np.ndarray:
    """Draw bounding boxes on a copy of the image."""
    out = image.copy()
    for r in ocr_results:
        pts = np.array(r["bbox"], dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
        x, y = pts[0]
        cv2.putText(
            out,
            r["text"][:30],
            (int(x), max(int(y) - 4, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 150, 0),
            1,
        )
    return out


def process_receipt(image_pil: Image.Image | None) -> tuple:
    if image_pil is None:
        return None, "{}", None

    img_rgb = np.array(image_pil.convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1].copy()

    proc = preprocess(img_bgr)
    ocr_results = read_image(proc)
    lines = extract_text_lines(ocr_results)
    receipt = extract(lines)

    # Overlay boxes on original RGB image
    annotated = _draw_boxes(img_rgb, ocr_results)
    annotated_pil = Image.fromarray(annotated)

    # Receipt JSON
    receipt_dict = receipt.model_dump()
    receipt_json = json.dumps(receipt_dict, indent=2, ensure_ascii=False)

    # Items DataFrame → CSV
    if receipt.items:
        df = pd.DataFrame([i.model_dump() for i in receipt.items])
    else:
        rows = [
            {
                "field": k,
                "value": v,
            }
            for k, v in receipt_dict.items()
            if k != "items"
        ]
        df = pd.DataFrame(rows)

    return annotated_pil, receipt_json, df


def build_demo() -> gr.Blocks:
    samples_dir = Path(__file__).parent / "data" / "samples"
    sample_images = sorted(samples_dir.glob("*.jpg")) + sorted(samples_dir.glob("*.png"))
    examples = [[str(p)] for p in sample_images[:3]] if sample_images else None

    with gr.Blocks(title="Receipt OCR Reader") as demo:
        gr.Markdown(
            "# Receipt OCR Reader\n"
            "Upload a receipt image — the model preprocesses it, runs EasyOCR, "
            "and extracts **merchant · date · total · line items** via rule-based parsing."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Receipt image")
                btn = gr.Button("Extract fields", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="OCR overlay (bounding boxes)")
                out_json = gr.Textbox(label="Extracted JSON", lines=12)
                out_df = gr.Dataframe(label="Fields / items")

        if examples:
            gr.Examples(examples=examples, inputs=inp)

        btn.click(process_receipt, inputs=inp, outputs=[out_img, out_json, out_df])

    return demo


if __name__ == "__main__":
    build_demo().launch()
