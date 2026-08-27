"""Tests for Phase 2 KPI and complaint-theme rules."""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

from ecommerce_rag.analytics import eda


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[1]")
        .appName("phase-2-eda-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_complaint_taxonomy_is_multilabel_and_has_fallbacks(
    spark: SparkSession,
) -> None:
    frame = spark.createDataFrame(
        [
            ("non-delivery", 1, "Ainda não recebi e quero reembolso", True),
            ("defect", 2, "Produto quebrado e com defeito", True),
            ("uncertain", 1, "ru", True),
            ("other", 2, "Embalagem sem manual adequado e instruções confusas", True),
            ("positive", 5, "Entrega perfeita", True),
        ],
        ["review_id", "review_score", "review_text", "text_is_eligible"],
    )

    classified = {
        row.review_id: set(row.complaint_themes)
        for row in eda.classify_complaint_themes(frame).collect()
    }

    assert classified["non-delivery"] == {"non_delivery", "service_or_refund"}
    assert classified["defect"] == {"damaged_or_defective"}
    assert classified["uncertain"] == {"uncertain"}
    assert classified["other"] == {"other"}
    assert "positive" not in classified


def test_monthly_rating_variation_uses_three_previous_months(
    spark: SparkSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eda, "MIN_GROUP_REVIEWS", 2)
    rows = []
    for month, score in [(1, 5), (2, 4), (3, 3), (4, 2)]:
        rows.extend([(2018, month, score, True), (2018, month, score, False)])
    reviews = spark.createDataFrame(
        rows,
        ["purchase_year", "purchase_month", "review_score", "text_is_eligible"],
    )

    april = (
        eda.build_monthly_review_statistics(reviews)
        .where("purchase_month = 4")
        .first()
    )

    assert april.baseline_month_count == 3
    assert april.baseline_average_rating == pytest.approx(4.0)
    assert april.average_rating == pytest.approx(2.0)
    assert april.rating_variation == pytest.approx(-2.0)
    assert april.negative_review_rate == pytest.approx(1.0)
