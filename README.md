# Receipt OCR Reader

Upload a photo of a receipt → OpenCV preprocessing → EasyOCR → rule-based field extraction
(merchant, date, total, line items) → validated JSON, served via FastAPI with a Gradio demo.

## Key results

Field-level accuracy on SROIE 2019 test set (361 receipts, rule-based extraction on ground-truth OCR text):

| Field | Correct | Total | Accuracy |
|---|---|---|---|
| Merchant (company) | 317 | 361 | **87.8 %** |
| Date | 345 | 361 | **95.6 %** |
| Total amount | 192 | 361 | **53.2 %** |

Numbers come from `reports/eval_gt_words.json`, committed.
Evaluation uses the dataset's own OCR text to isolate extraction accuracy from OCR quality.
The date parser handles English, Italian (gg/mm/aaaa), and 2-digit-year formats.
Total accuracy is lower because Malaysian receipts contain multiple amounts (subtotal, tax,
cash given, change) — the most common failure mode is picking the cash/change line instead
of the net total.

## Story: Pinfopedia → this

At Pinfopedia I built QR-code labels that linked physical products to digital pages — turning
something you can hold into a structured record. This project continues that thread:
point a camera at a receipt, get back a structured JSON object.
Rule-based parsing, not an LLM, keeps it deterministic and auditable — no hallucinated totals.

## Architecture

```
image ─► OpenCV preprocess ─► EasyOCR ─► text + boxes
          (deskew, denoise,                    │
           binarise)           field extraction (regex/rules)
                                               │
                         merchant · date · total · items
                                               │
                    FastAPI POST /extract  ─►  Receipt JSON
                    Gradio demo            ─►  overlay + table + CSV export
```

## Stack

Python 3.11 · EasyOCR · OpenCV-headless · pydantic v2 · FastAPI · Gradio · pandas · pytest · ruff

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install easyocr opencv-python-headless pydantic fastapi "uvicorn[standard]" \
    gradio pandas pillow python-multipart
pip install -e .

# API server (http://localhost:8000)
uvicorn api.main:app --reload

# Gradio demo (http://localhost:7860)
python app.py

# Extract from a single receipt image (cURL)
curl -F "file=@receipt.jpg" http://localhost:8000/extract
```

## Run evaluation

```bash
# Download SROIE 2019 test set (361 receipts from HuggingFace)
pip install datasets
python eval/download_sroie.py

# Fast field-extraction eval (uses dataset's own OCR text — no EasyOCR needed)
python eval/run_eval.py

# Full end-to-end eval (runs EasyOCR on images — slow on CPU)
python eval/run_eval.py --ocr --limit 50
```

## Tests

```bash
pytest -q   # 36 passed
```

- `test_schema.py` — pydantic Receipt validation (6 tests)
- `test_extract.py` — date/amount/merchant extraction incl. Italian formats (22 tests)
- `test_preprocess.py` — OpenCV pipeline shapes (5 tests)
- `test_api.py` — FastAPI contract, non-image rejection, mock OCR (4 tests)

## Error analysis (where it fails)

**Total (53 %)** — Most failures are receipts where "TOTAL" appears on a label-only line with
the amount on the next line, or where multiple amounts (subtotal, rounding, cash) precede the
real total. A layout-aware model (Donut, LayoutLM) would handle these structurally.

**Merchant (12 % wrong)** — Short company names where the first non-trivial text line is a
receipt-header code rather than the store name. Requires top-of-page layout heuristics.

**Date (4 % wrong)** — Mostly receipts with non-standard date strings (e.g., timestamp
inside a longer line where the time portion confuses the day/month pattern).

## Dataset & licence

SROIE 2019 (ICDAR2019 Scanned Receipts OCR & Information Extraction) — research dataset;
see `eval/DATASET.md`. EasyOCR (Apache-2.0). Code: MIT — see [LICENSE](LICENSE).
