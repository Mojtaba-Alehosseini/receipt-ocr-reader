"""Rule-based field extractor tests — no OCR needed."""

import pytest

from ocr.extract import (
    _normalise_amount,
    extract,
    extract_date,
    extract_merchant,
    extract_total,
)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
class TestExtractDate:
    def test_iso_format(self):
        assert extract_date(["Receipt 2024-06-15"]) == "2024-06-15"

    def test_dmy_slash(self):
        assert extract_date(["Date: 15/06/2024"]) == "2024-06-15"

    def test_dmy_dot_italian(self):
        assert extract_date(["Data: 15.06.2024"]) == "2024-06-15"

    def test_dmy_dash(self):
        assert extract_date(["15-06-2024"]) == "2024-06-15"

    def test_named_month_english(self):
        result = extract_date(["15 Jun 2024"])
        assert result == "2024-06-15"

    def test_no_date_returns_none(self):
        assert extract_date(["no date here", "just text"]) is None

    def test_multiple_lines_picks_first(self):
        lines = ["SUPERMERCATO", "15/06/2024", "22/07/2024"]
        result = extract_date(lines)
        assert result == "2024-06-15"


# ---------------------------------------------------------------------------
# Amount normalisation
# ---------------------------------------------------------------------------
class TestNormaliseAmount:
    def test_dot_decimal(self):
        assert _normalise_amount("12.50") == pytest.approx(12.50)

    def test_comma_decimal_italian(self):
        assert _normalise_amount("12,50") == pytest.approx(12.50)

    def test_thousands_dot_decimal_comma(self):
        assert _normalise_amount("1.234,56") == pytest.approx(1234.56)

    def test_thousands_comma_decimal_dot(self):
        assert _normalise_amount("1,234.56") == pytest.approx(1234.56)


# ---------------------------------------------------------------------------
# Total extraction
# ---------------------------------------------------------------------------
class TestExtractTotal:
    def test_total_keyword_english(self):
        lines = ["Milk 1.50", "Bread 2.00", "Total 3.50"]
        assert extract_total(lines) == pytest.approx(3.50)

    def test_totale_keyword_italian(self):
        lines = ["Latte 1,50", "Pane 2,00", "Totale 3,50"]
        assert extract_total(lines) == pytest.approx(3.50)

    def test_fallback_last_amount(self):
        # No total keyword — falls back to last non-trivial amount
        lines = ["Item 1.00", "Other 5.99", "Something 2.50"]
        assert extract_total(lines) == pytest.approx(2.50)

    def test_no_amounts_returns_none(self):
        assert extract_total(["just text", "no numbers"]) is None


# ---------------------------------------------------------------------------
# Merchant extraction
# ---------------------------------------------------------------------------
class TestExtractMerchant:
    def test_first_clean_line(self):
        assert extract_merchant(["SUPERMERCATO ROSSI", "Via Roma 1"]) == "SUPERMERCATO ROSSI"

    def test_skips_short_lines(self):
        assert extract_merchant(["AB", "MERCHANT NAME"]) == "MERCHANT NAME"

    def test_skips_pure_numbers(self):
        assert extract_merchant(["123456", "ACME STORE"]) == "ACME STORE"

    def test_returns_none_when_no_candidate(self):
        assert extract_merchant(["12", "45"]) is None


# ---------------------------------------------------------------------------
# Full extract pipeline
# ---------------------------------------------------------------------------
class TestExtract:
    def test_full_receipt(self):
        lines = [
            "SUPERMERCATO ROSSI",
            "Via Roma 1, Milano",
            "Data: 15/06/2024",
            "Milk           1.50",
            "Bread          2.00",
            "Total          3.50",
        ]
        r = extract(lines)
        assert r.merchant == "SUPERMERCATO ROSSI"
        assert r.date == "2024-06-15"
        assert r.total == pytest.approx(3.50)

    def test_empty_lines(self):
        r = extract([])
        assert r.merchant is None
        assert r.total is None
