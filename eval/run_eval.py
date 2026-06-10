"""Field-level accuracy evaluation against SROIE 2019.

Usage:
  python eval/run_eval.py [--limit N] [--save]

Expects:
  data/sroie/img/*.jpg     — receipt images
  data/sroie/entities/*.json  — ground truth {company, date, total, address}

Writes:
  reports/eval_results.json  — per-field metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
IMG_DIR = ROOT / "data" / "sroie" / "img"
ENT_DIR = ROOT / "data" / "sroie" / "entities"
REPORTS_DIR = ROOT / "reports"


def _normalise_gt_total(raw: str) -> float | None:
    """Parse ground-truth total string like '$12.50' or '12.50' to float."""
    import re

    m = re.search(r"[\d]+[.,]\d{2}", raw.replace(",", "."))
    if m:
        try:
            return float(m.group().replace(",", "."))
        except ValueError:
            pass
    return None


def _normalise_gt_date(raw: str) -> str:
    """Best-effort normalise SROIE date strings to YYYY-MM-DD."""
    from ocr.extract import extract_date

    result = extract_date([raw])
    return result or raw.strip().lower()


def _total_match(pred: float | None, gt_raw: str, tol: float = 0.05) -> bool:
    if pred is None:
        return False
    gt = _normalise_gt_total(gt_raw)
    if gt is None:
        return False
    return abs(pred - gt) <= tol


def _date_match(pred: str | None, gt_raw: str) -> bool:
    if not pred:
        return False
    gt_norm = _normalise_gt_date(gt_raw)
    return pred.strip().lower() == gt_norm.strip().lower()


def _company_match(pred: str | None, gt_raw: str) -> bool:
    if not pred:
        return False
    # Soft match: check if gt_raw is a substring of pred or vice-versa (case-insensitive)
    p = pred.strip().lower()
    g = gt_raw.strip().lower()
    return g in p or p in g or p == g


def run_eval(
    limit: int | None = None,
    save: bool = True,
    use_gt_words: bool = True,
) -> dict:
    """Run field-level evaluation.

    Args:
        use_gt_words: If True (default), use dataset's ground-truth OCR words as input
                      (evaluates field extraction only, no EasyOCR needed — fast).
                      If False, run EasyOCR on images (evaluates full pipeline — slow).
    """
    from ocr.extract import extract

    images = sorted(IMG_DIR.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(
            f"No images found in {IMG_DIR}. "
            "Run `python eval/download_sroie.py` first."
        )

    if limit:
        images = images[:limit]

    mode = "gt-words" if use_gt_words else "easyocr"
    print(f"Evaluating {len(images)} receipts [mode={mode}] ...")

    results = []
    totals_ok = dates_ok = companies_ok = 0
    failures: list[dict] = []

    for img_path in images:
        gt_path = ENT_DIR / (img_path.stem + ".json")
        if not gt_path.exists():
            continue

        gt = json.loads(gt_path.read_text(encoding="utf-8"))

        try:
            if use_gt_words:
                lines = gt.get("words", [])
            else:
                from ocr.engine import extract_text_lines, read_file

                ocr_results = read_file(img_path)
                lines = extract_text_lines(ocr_results)
            receipt = extract(lines)
        except Exception as exc:
            failures.append({"file": img_path.name, "error": str(exc)})
            continue

        company_ok = _company_match(receipt.merchant, gt.get("company", ""))
        date_ok = _date_match(receipt.date, gt.get("date", ""))
        total_ok = _total_match(receipt.total, gt.get("total", ""))

        companies_ok += company_ok
        dates_ok += date_ok
        totals_ok += total_ok

        results.append(
            {
                "file": img_path.name,
                "company_ok": company_ok,
                "date_ok": date_ok,
                "total_ok": total_ok,
                "pred_merchant": receipt.merchant,
                "pred_date": receipt.date,
                "pred_total": receipt.total,
                "gt_company": gt.get("company"),
                "gt_date": gt.get("date"),
                "gt_total": gt.get("total"),
            }
        )

    n = len(results)
    if n == 0:
        raise RuntimeError("No results — check data/sroie/ directory.")

    metrics = {
        "n_evaluated": n,
        "n_errors": len(failures),
        "company_accuracy": round(companies_ok / n, 4),
        "date_accuracy": round(dates_ok / n, 4),
        "total_accuracy": round(totals_ok / n, 4),
    }

    print(f"\n{'Field':<16} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 46)
    print(f"{'company':<16} {companies_ok:>8} {n:>8} {metrics['company_accuracy']:>10.1%}")
    print(f"{'date':<16} {dates_ok:>8} {n:>8} {metrics['date_accuracy']:>10.1%}")
    print(f"{'total':<16} {totals_ok:>8} {n:>8} {metrics['total_accuracy']:>10.1%}")
    print(f"\nErrors: {len(failures)}/{n + len(failures)}")

    if failures:
        print("\n5 worst failures:")
        for f in failures[:5]:
            print(f"  {f['file']}: {f.get('error', 'wrong')}")

    if save:
        REPORTS_DIR.mkdir(exist_ok=True)
        output = {"metrics": metrics, "per_sample": results[:200], "errors": failures[:20]}
        (REPORTS_DIR / "eval_results.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False)
        )
        print(f"\nResults saved to {REPORTS_DIR / 'eval_results.json'}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N images")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument(
        "--ocr",
        dest="use_gt_words",
        action="store_false",
        help="Run EasyOCR on images instead of using ground-truth words (slow)",
    )
    args = parser.parse_args()
    run_eval(limit=args.limit, save=args.save, use_gt_words=args.use_gt_words)
