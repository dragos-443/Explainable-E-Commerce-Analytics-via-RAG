"""Shared eligibility rule for review text used by the RAG pipeline."""

from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F


def has_meaningful_text(text: Column) -> Column:
    """Return true when text contains at least one alphabetic token of length two."""
    normalized = F.trim(
        F.regexp_replace(
            F.lower(F.coalesce(text, F.lit(""))),
            r"[^\p{L}]+",
            " ",
        )
    )
    tokens = F.split(normalized, r"\s+")
    return F.exists(tokens, lambda token: F.length(token) >= 2)


def rag_text_is_eligible() -> Column:
    """Combine source-field availability with minimum RAG text quality."""
    return F.col("text_is_eligible") & has_meaningful_text(F.col("review_text"))
