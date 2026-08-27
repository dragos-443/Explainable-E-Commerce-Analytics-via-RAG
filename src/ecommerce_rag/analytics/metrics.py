"""Central definitions shared by the analytical components."""

from __future__ import annotations

from typing import Dict

from pyspark.sql import Column
from pyspark.sql import functions as F


NEGATIVE_SCORE_MAX = 2
LOW_RATING_SCORES = (1, 2)
BASELINE_MONTH_COUNT = 3
METRIC_DEFINITION_VERSION = "1.0"


KPI_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "average_rating": {
        "role": "phenomenon",
        "formula": "sum(review_score) / reviews with a valid score",
        "unit": "rating_points",
    },
    "rating_variation": {
        "role": "phenomenon",
        "formula": "current average_rating - baseline average_rating",
        "unit": "rating_points",
    },
    "negative_review_rate": {
        "role": "phenomenon",
        "formula": "reviews with review_score <= 2 / reviews with a valid score",
        "unit": "proportion",
    },
    "low_rating_distribution": {
        "role": "phenomenon",
        "formula": "reviews with score 1 or 2 by score / reviews with a valid score",
        "unit": "proportion",
    },
    "late_delivery_rate": {
        "role": "supporting_evidence",
        "formula": "late orders / orders with observed and estimated delivery dates",
        "unit": "proportion",
    },
    "average_delivery_delay": {
        "role": "supporting_evidence",
        "formula": "average delivery_delay_days over late orders",
        "unit": "days",
    },
    "order_status_distribution": {
        "role": "supporting_evidence",
        "formula": "orders by status / filtered orders",
        "unit": "proportion",
    },
    "order_volume": {
        "role": "contextual",
        "formula": "count of distinct orders in the selected period",
        "unit": "orders",
    },
    "average_order_value": {
        "role": "contextual",
        "formula": "average order_value over orders with a known value",
        "unit": "BRL",
    },
}


def negative_indicator(column: str = "review_score") -> Column:
    """Return 1 for a negative review and 0 for another valid score."""
    return F.when(F.col(column) <= NEGATIVE_SCORE_MAX, 1).otherwise(0)
