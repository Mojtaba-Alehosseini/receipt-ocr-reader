"""Rule-based field extraction from OCR text lines.

Extracts: merchant, date, total, line items.
Handles English and Italian date/number formats deterministically.
"""

from __future__ import annotations

import re
from typing import Optional

from ocr.schema import LineItem, Receipt

# ---------------------------------------------------------------------------
# Date patterns — ordered most-specific to least
# ---------------------------------------------------------------------------
# Italian/European: gg/mm/aaaa  or  gg-mm-aaaa  or  gg.mm.aaaa  (4-digit year)
_DATE_DMY = re.compile(
    r"\b(?P<d>\d{1,2})[/\-\.](?P<m>\d{1,2})[/\-\.](?P<y>\d{4})\b"
)
# Short 2-digit year: DD/MM/YY or DD-MM-YY or DD.MM.YY
_DATE_DMY2 = re.compile(
    r"\b(?P<d>\d{1,2})[/\-\.](?P<m>\d{1,2})[/\-\.](?P<y>\d{2})\b"
)
# ISO: yyyy-mm-dd
_DATE_ISO = re.compile(r"\b(?P<y>\d{4})[/\-\.](?P<m>\d{1,2})[/\-\.](?P<d>\d{1,2})\b")
# Month-name English: Jan 12 2024 / 12 Jan 2024 / 12 Jan 18 (2-digit year)
_MONTH_NAMES = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    "january|february|march|april|june|july|august|september|october|november|december"
)
_DATE_NAMED = re.compile(
    rf"\b(\d{{1,2}})\s+({_MONTH_NAMES})[,\s]+(\d{{2,4}})\b"
    rf"|\b({_MONTH_NAMES})\s+(\d{{1,2}})[,\s]+(\d{{2,4}})\b",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "jan": "01", "january": "01", "feb": "02", "february": "02",
    "mar": "03", "march": "03", "apr": "04", "april": "04",
    "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "september": "09",
    "oct": "10", "october": "10", "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

# ---------------------------------------------------------------------------
# Amount patterns — handle decimal comma (Italian) and decimal dot
# ---------------------------------------------------------------------------
# Matches: 12.50  12,50  12,500.00  €12.50  $12.50  etc.
_AMOUNT = re.compile(
    r"(?:[$€£])\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2}))"
    r"|(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})\b"
)

# Total keyword patterns — ordered by specificity (most specific first)
_TOTAL_KWS = re.compile(
    r"\b(grand\s+total|net\s+total|total\s+amount|total\s+payable|total\s+due"
    r"|amount\s+due|amount\s+payable|balance\s+due|balance\s+payable"
    r"|total[ei]?|subtotal|totale|da\s+pagare|importo)\b",
    re.IGNORECASE,
)

# Line-item pattern: text followed by an amount (or amount then text)
_ITEM_LINE = re.compile(
    r"^(?P<desc>.+?)\s+(?P<amt>\d{1,3}(?:[.,]\d{3})*[.,]\d{2})\s*$"
    r"|^(?P<amt2>\d{1,3}(?:[.,]\d{3})*[.,]\d{2})\s+(?P<desc2>.+?)\s*$"
)


def _normalise_amount(raw: str) -> float:
    """Convert '12,50' or '12.50' or '1,234.56' to float."""
    raw = raw.strip()
    # If both comma and dot present: last separator is decimal
    if "," in raw and "." in raw:
        # e.g. 1,234.56 → remove comma thousands-separator
        if raw.rfind(".") > raw.rfind(","):
            raw = raw.replace(",", "")
        else:
            # e.g. 1.234,56 (European)
            raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        # Could be thousands (1,000) or decimal (12,50)
        parts = raw.split(",")
        if len(parts) == 2 and len(parts[1]) == 2:
            raw = raw.replace(",", ".")
        else:
            raw = raw.replace(",", "")
    return float(raw)


def _normalise_date(year: str, month: str, day: str) -> str:
    try:
        y, m, d = int(year), int(month), int(day)
        if y < 100:
            y += 2000  # 2-digit year: 18 → 2018
        return f"{y:04d}-{m:02d}-{d:02d}"
    except ValueError:
        return f"{year}-{month}-{day}"


def extract_date(lines: list[str]) -> Optional[str]:
    for line in lines:
        m = _DATE_ISO.search(line)
        if m:
            return _normalise_date(m.group("y"), m.group("m"), m.group("d"))
        m = _DATE_DMY.search(line)
        if m:
            d, mo, y = m.group("d"), m.group("m"), m.group("y")
            if int(d) <= 31 and int(mo) <= 12:
                return _normalise_date(y, mo, d)
        m = _DATE_DMY2.search(line)
        if m:
            d, mo, y = m.group("d"), m.group("m"), m.group("y")
            # Exclude pure time patterns like 11:05 (no slash/dot/dash) — already handled
            # Only match if d ≤ 31 and mo ≤ 12
            if int(d) <= 31 and int(mo) <= 12:
                return _normalise_date(y, mo, d)
        m = _DATE_NAMED.search(line)
        if m:
            groups = m.groups()
            if groups[0]:  # DD MonthName YY/YYYY
                day_, mon_name, year_ = groups[0], groups[1].lower(), groups[2]
                return _normalise_date(year_, _MONTH_MAP.get(mon_name, mon_name), day_)
            else:
                mon_name, day_, year_ = groups[3].lower(), groups[4], groups[5]
                return _normalise_date(year_, _MONTH_MAP.get(mon_name, mon_name), day_)
    return None


def _all_amounts(text: str) -> list[float]:
    amounts = []
    for m in _AMOUNT.finditer(text):
        raw = m.group(1) or m.group(2)
        if raw:
            try:
                amounts.append(_normalise_amount(raw))
            except ValueError:
                pass
    return amounts


def extract_total(lines: list[str]) -> Optional[float]:
    # Priority 1: line that mentions a total keyword + has an amount on the SAME line
    for i, line in enumerate(lines):
        if _TOTAL_KWS.search(line):
            amounts = _all_amounts(line)
            if amounts:
                return max(amounts)
            # Try next line (keyword and amount sometimes split across lines)
            if i + 1 < len(lines):
                amounts = _all_amounts(lines[i + 1])
                if amounts:
                    return max(amounts)

    # Priority 2: last non-trivial amount in the bottom half of the receipt
    # (avoids picking CASH/CHANGE amounts that often appear after the total)
    _PAYMENT_KWS = re.compile(
        r"\b(cash|change|paid|visa|master|card|rounding|gst|tax|disc|discount|credit)\b",
        re.IGNORECASE,
    )
    bottom = lines[len(lines) // 2 :]
    for line in reversed(bottom):
        if _PAYMENT_KWS.search(line):
            continue
        amounts = [a for a in _all_amounts(line) if a > 0.0]
        if amounts:
            return amounts[0]

    # Priority 3: any non-zero amount anywhere
    for line in reversed(lines):
        amounts = [a for a in _all_amounts(line) if a > 0.0]
        if amounts:
            return amounts[0]

    return None


def extract_merchant(lines: list[str]) -> Optional[str]:
    """Return first non-trivial text line as merchant name."""
    for line in lines[:5]:
        stripped = line.strip()
        # Skip very short lines, pure numbers, or lines that look like dates/totals
        if len(stripped) < 3:
            continue
        if re.fullmatch(r"[\d\s/\-\.:]+", stripped):
            continue
        if _TOTAL_KWS.search(stripped):
            continue
        return stripped
    return None


def extract_items(lines: list[str]) -> list[LineItem]:
    items: list[LineItem] = []
    for line in lines:
        if _TOTAL_KWS.search(line):  # skip total/subtotal summary lines
            continue
        m = _ITEM_LINE.match(line.strip())
        if m:
            if m.group("desc") and m.group("amt"):
                try:
                    items.append(
                        LineItem(
                            description=m.group("desc").strip(),
                            amount=_normalise_amount(m.group("amt")),
                        )
                    )
                except (ValueError, Exception):
                    pass
            elif m.group("desc2") and m.group("amt2"):
                try:
                    items.append(
                        LineItem(
                            description=m.group("desc2").strip(),
                            amount=_normalise_amount(m.group("amt2")),
                        )
                    )
                except (ValueError, Exception):
                    pass
    return items


def validate_total(
    total: Optional[float],
    items: list[LineItem],
    tolerance: float = 0.05,
) -> bool:
    """Return True if the sum of item amounts matches the declared total within tolerance.

    Useful as a post-extraction sanity check: if the OCR parsed the items but the
    line-by-line sum deviates from the total field by more than `tolerance` (absolute),
    the extraction is likely incomplete or mis-parsed.
    """
    if total is None or not items:
        return False
    items_sum = sum(item.amount for item in items)
    return abs(items_sum - total) <= tolerance


def extract(lines: list[str]) -> Receipt:
    """Run all extractors and return a validated Receipt."""
    return Receipt(
        merchant=extract_merchant(lines),
        date=extract_date(lines),
        total=extract_total(lines),
        items=extract_items(lines),
    )
