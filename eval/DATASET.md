# SROIE 2019 dataset

The evaluation script uses the SROIE 2019 (Scanned Receipts OCR & Information Extraction)
dataset from ICDAR 2019.

## Download

```bash
pip install datasets
python eval/download_sroie.py
```

This places images under `data/sroie/img/` and ground-truth JSON under `data/sroie/entities/`.
`data/sroie/` is gitignored; results written to `reports/eval_results.json`.

## License
SROIE is a research dataset from ICDAR 2019.  Refer to the original competition for licence terms.
HuggingFace dataset id: `jsdnrs/ICDAR2019-SROIE`.
