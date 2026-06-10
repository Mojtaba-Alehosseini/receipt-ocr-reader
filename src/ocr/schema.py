from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator


class LineItem(BaseModel):
    description: str
    amount: Optional[float] = None


class Receipt(BaseModel):
    merchant: Optional[str] = None
    date: Optional[str] = None  # normalised to YYYY-MM-DD when possible
    total: Optional[float] = None
    items: list[LineItem] = []

    @field_validator("total")
    @classmethod
    def total_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("total must be non-negative")
        return v
