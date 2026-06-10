"""Download SROIE 2019 into data/sroie/ using the HuggingFace datasets library."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
IMG_DIR = ROOT / "data" / "sroie" / "img"
ENT_DIR = ROOT / "data" / "sroie" / "entities"


def download() -> None:
    from datasets import load_dataset

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ENT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading SROIE from HuggingFace (jsdnrs/ICDAR2019-SROIE)...")
    ds = load_dataset("jsdnrs/ICDAR2019-SROIE", split="test")

    for i, row in enumerate(ds):
        stem = f"{i:04d}"
        # Save image
        row["image"].save(IMG_DIR / f"{stem}.jpg")
        # Save ground-truth fields (nested under "entities") + words for fast eval
        ent = row.get("entities", {})
        entities = {
            "company": ent.get("company", ""),
            "date": ent.get("date", ""),
            "address": ent.get("address", ""),
            "total": ent.get("total", ""),
            "words": row.get("words", []),  # ground-truth OCR words for fast eval
        }
        (ENT_DIR / f"{stem}.json").write_text(json.dumps(entities, ensure_ascii=False))

    print(f"Saved {len(ds)} receipts to {IMG_DIR} and {ENT_DIR}")


if __name__ == "__main__":
    download()
