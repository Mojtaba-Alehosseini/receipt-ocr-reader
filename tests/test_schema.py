"""pydantic Receipt schema validation tests."""

import pytest
from pydantic import ValidationError

from ocr.schema import LineItem, Receipt


def test_receipt_empty():
    r = Receipt()
    assert r.merchant is None
    assert r.total is None
    assert r.items == []


def test_receipt_valid():
    r = Receipt(merchant="Supermercato", date="2024-01-15", total=42.50, items=[])
    assert r.total == 42.50


def test_receipt_with_items():
    r = Receipt(
        merchant="Shop",
        items=[LineItem(description="Apple", amount=0.99), LineItem(description="Bread")],
    )
    assert len(r.items) == 2
    assert r.items[1].amount is None


def test_receipt_negative_total_rejected():
    with pytest.raises(ValidationError):
        Receipt(total=-1.0)


def test_line_item_no_amount():
    item = LineItem(description="Coffee")
    assert item.amount is None


def test_receipt_json_round_trip():
    r = Receipt(merchant="ALDI", date="2024-06-01", total=12.30)
    data = r.model_dump()
    r2 = Receipt.model_validate(data)
    assert r == r2
