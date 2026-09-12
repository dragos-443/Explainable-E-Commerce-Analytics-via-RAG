"""Integration checks for real Phase 5 grounded explanation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


OUTPUT_ROOT = Path("/workspace/reports/integration/phase5")


def test_march_rating_explanation_is_grounded_end_to_end() -> None:
    payload = json.loads(
        (OUTPUT_ROOT / "march-rating-drop.json").read_text(encoding="utf-8")
    )
    assert payload["question"]["intent"] == "explain_rating_drop"
    assert payload["question"]["start_month"] == "2018-03"
    assert payload["answer_language"] == "it"
    assert payload["insufficient_evidence"]["is_insufficient"] is False
    assert payload["analytics"]["metrics"]["rating_variation"]["value"] < 0
    assert payload["theme_evidence"]["baseline_filters"] is not None
    population = payload["theme_evidence"]["current_population"]
    assert population["negative_text_reviews"] > 0
    assert 0 < population["negative_text_coverage"] <= 1
    for theme in payload["theme_evidence"]["themes"]:
        if theme["current_mention_rate"] is not None:
            assert theme["current_mention_rate"] == pytest.approx(
                theme["current_mentions"] / population["negative_text_reviews"]
            )
    assert payload["generation"]["generation_status"] in {
        "llm_generated_validated",
        "validated_fallback",
    }
    interpretation = payload["generation"]["interpretation"]
    assert len(interpretation) >= 100
    assert sum(interpretation.count(mark) for mark in ".!?") >= 2
    assert "perché" not in interpretation.lower()
    assert payload["generation"]["evidence_limit"] == (
        "Le evidenze sono descrittive e non dimostrano un rapporto causale "
        "né rappresentano tutta la popolazione."
    )
    allowed_ids = {
        item["review_id"] for item in payload["context"]["review_evidence"]
    }
    assert set(payload["generation"]["review_ids"]).issubset(allowed_ids)
    assert payload["generation"]["review_ids"]
    assert all(item["document_original"] for item in payload["context"]["review_evidence"])
    assert all(item["translation"] for item in payload["context"]["review_evidence"])
    assert all(review_id in payload["answer_it"] for review_id in allowed_ids)
    assert "top-k" in payload["answer_it"]


def test_descriptive_metric_reports_insufficient_review_evidence() -> None:
    payload = json.loads(
        (OUTPUT_ROOT / "order-volume.json").read_text(encoding="utf-8")
    )
    assert payload["question"]["intent"] == "descriptive_only"
    assert payload["question"]["metric"] == "order_volume"
    assert payload["insufficient_evidence"]["is_insufficient"] is True
    assert payload["generation"] is None
    assert payload["context"]["review_evidence"] == []
    assert "Evidenza insufficiente" in payload["answer_it"]
