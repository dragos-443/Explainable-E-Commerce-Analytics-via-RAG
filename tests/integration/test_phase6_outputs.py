"""Integration checks for the real, reproducible Phase 6 demo suite."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT_ROOT = Path("/workspace/reports/demo/phase6")


def _load(case_id: str) -> dict:
    return json.loads((OUTPUT_ROOT / f"{case_id}.json").read_text(encoding="utf-8"))


def test_demo_summary_contains_four_verified_scenarios() -> None:
    summary = _load("summary")
    assert summary["case_count"] == 4
    assert summary["passed_count"] == 4
    assert summary["all_passed"] is True
    assert {case["kind"] for case in summary["cases"]} == {
        "logistics",
        "product_issues",
        "review_analysis",
        "insufficient_evidence",
    }
    assert all(case["passed"] and not case["failed_checks"] for case in summary["cases"])


def test_logistics_case_connects_rating_drop_to_supported_logistics_hypotheses() -> None:
    payload = _load("logistics_march_2018")
    metrics = payload["analytics"]["metrics"]
    themes = {
        item["theme"] for item in payload["theme_evidence"]["ranked_hypotheses"][:3]
    }
    assert payload["demo_verification"]["passed"] is True
    assert metrics["rating_variation"]["value"] < 0
    assert metrics["late_delivery_rate"]["change"] > 0
    assert {"non_delivery", "delivery_delay"}.issubset(themes)


def test_product_case_separates_product_complaints_from_delivery_performance() -> None:
    payload = _load("office_furniture_product_issues")
    comparison = payload["demo_verification"]["comparison"]
    themes = {
        item["theme"] for item in payload["theme_evidence"]["ranked_hypotheses"][:3]
    }
    assert payload["demo_verification"]["passed"] is True
    assert comparison["rating_difference_vs_overall"] < 0
    assert abs(comparison["late_delivery_rate_difference_vs_overall"]) <= 0.02
    assert "wrong_or_missing_item" in themes


def test_qualitative_cases_keep_original_translation_and_grounding() -> None:
    for case_id in (
        "logistics_march_2018",
        "office_furniture_product_issues",
        "overall_complaint_analysis",
    ):
        payload = _load(case_id)
        evidence = payload["context"]["review_evidence"]
        allowed_ids = {item["review_id"] for item in evidence}
        assert payload["generation"]["generation_status"] in {
            "llm_generated_validated",
            "validated_fallback",
        }
        assert len(evidence) >= 3
        assert all(item["document_original"] for item in evidence)
        assert all(item["translation"]["status"] == "translated" for item in evidence)
        assert all(
            item["metadata"][f"theme_{item['retrieved_for_theme']}"]
            for item in evidence
        )
        assert set(payload["generation"]["review_ids"]).issubset(allowed_ids)
        assert all(review_id in payload["answer_it"] for review_id in allowed_ids)


def test_volume_case_stops_when_reviews_cannot_explain_the_metric() -> None:
    payload = _load("order_volume_insufficient")
    assert payload["question"]["intent"] == "descriptive_only"
    assert payload["question"]["metric"] == "order_volume"
    assert payload["demo_verification"]["passed"] is True
    assert payload["insufficient_evidence"]["is_insufficient"] is True
    assert payload["generation"] is None
    assert payload["context"]["review_evidence"] == []
