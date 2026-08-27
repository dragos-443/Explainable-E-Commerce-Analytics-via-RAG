"""Validate structured outputs produced by the Phase 3 engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


OUTPUT_ROOT = Path("/workspace/reports/analytics/phase3")


@pytest.mark.parametrize(
    "filename,expected_filters",
    [
        ("overall.json", {}),
        ("march-2018.json", {"start_month": "2018-03", "end_month": "2018-03"}),
        ("office-furniture.json", {"product_category": "office_furniture"}),
    ],
)
def test_real_query_output_contract(filename: str, expected_filters: dict) -> None:
    payload = json.loads((OUTPUT_ROOT / filename).read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["metric_definition_version"] == "1.0"
    assert payload["interpretation_limit"]
    assert set(payload["metrics"]) == {
        "average_rating",
        "rating_variation",
        "negative_review_rate",
        "low_rating_distribution",
        "late_delivery_rate",
        "average_delivery_delay",
        "order_status_distribution",
        "order_volume",
        "average_order_value",
    }
    for key, value in expected_filters.items():
        assert payload["filters"][key] == value
    assert payload["populations"]["reviews"] > 0
    assert payload["populations"]["orders"] > 0


def test_march_query_contains_expected_deterioration() -> None:
    payload = json.loads((OUTPUT_ROOT / "march-2018.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]

    assert metrics["rating_variation"]["value"] == pytest.approx(-0.2096, abs=0.001)
    assert metrics["negative_review_rate"]["value"] > metrics["negative_review_rate"]["baseline_value"]
    assert metrics["late_delivery_rate"]["value"] > metrics["late_delivery_rate"]["baseline_value"]
