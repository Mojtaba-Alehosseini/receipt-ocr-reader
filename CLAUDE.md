# Project: Receipt / product-label OCR reader

## What this is
Image → OCR (EasyOCR) → rule-based field extraction (merchant, date, total, line items) → JSON/CSV,
served via FastAPI with a Gradio demo, evaluated at field level against SROIE 2019. Continues my
Pinfopedia QR-label work (physical → structured digital data).

## Stack
Python 3.11 · EasyOCR + OpenCV · regex/rule parser · pydantic v2 · FastAPI · Gradio · pandas · pytest · ruff.

## Commands
- API: `uvicorn api.main:app --reload`
- Demo: `python app.py`
- Eval: `python eval/run_eval.py`
- Tests: `pytest -q`
- Lint: `ruff check .`

## Conventions
- Parsing is rule-based and deterministic by default. Handle Italian date/number formats.
- Output validates against the pydantic Receipt schema.
- Use `opencv-python-headless` (not `opencv-python`) — no display needed.
- EasyOCR is the engine (not PaddleOCR — PaddlePaddle is painful on Windows).
- Don't commit the full SROIE set — data/sroie/ is gitignored; see eval/DATASET.md for download.
- src/ layout: `pip install -e .`, modules as `from ocr.engine import read_image`.
- ROOT = Path(__file__).parents[2] for files at src/ocr/*.py.

## Done per step
Verify command passes → commit.
