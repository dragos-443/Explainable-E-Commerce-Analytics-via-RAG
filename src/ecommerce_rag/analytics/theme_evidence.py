"""Spark aggregation of complaint themes over complete review populations."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ecommerce_rag.analytics.engine import (
    AnalyticsFilters,
    apply_filters,
    baseline_filters,
)
from ecommerce_rag.analytics.metrics import NEGATIVE_SCORE_MAX
from ecommerce_rag.rag.text_quality import rag_text_is_eligible
from ecommerce_rag.rag.themes import ALL_THEMES, FALLBACK_THEMES


THEME_METRIC_VERSION = "1.0"


def _safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _population_summary(reviews: DataFrame) -> Dict[str, Any]:
    row = reviews.agg(
        F.count("review_score").alias("total_reviews"),
        F.coalesce(
            F.sum(F.when(rag_text_is_eligible(), 1).otherwise(0)), F.lit(0)
        ).alias("text_reviews"),
        F.coalesce(
            F.sum(
                F.when(F.col("review_score") <= NEGATIVE_SCORE_MAX, 1).otherwise(0)
            ),
            F.lit(0),
        ).alias("negative_reviews"),
        F.coalesce(
            F.sum(
                F.when(
                    (F.col("review_score") <= NEGATIVE_SCORE_MAX)
                    & rag_text_is_eligible(),
                    1,
                ).otherwise(0)
            ),
            F.lit(0),
        ).alias("negative_text_reviews"),
    ).first()
    values = row.asDict()
    values["overall_text_coverage"] = _safe_rate(
        values["text_reviews"], values["total_reviews"]
    )
    values["negative_text_coverage"] = _safe_rate(
        values["negative_text_reviews"], values["negative_reviews"]
    )
    return values


def _theme_counts(reviews: DataFrame, review_themes: DataFrame) -> Dict[str, int]:
    eligible_negative_ids = reviews.where(
        (F.col("review_score") <= NEGATIVE_SCORE_MAX) & rag_text_is_eligible()
    ).select("review_id")
    return {
        row.theme: row["count"]
        for row in eligible_negative_ids.join(
            review_themes.select("review_id", "theme"), "review_id", "inner"
        )
        .groupBy("theme")
        .agg(F.countDistinct("review_id").alias("count"))
        .collect()
    }


class ThemeEvidenceEngine:
    """Quantify multi-label themes without estimating prevalence from Chroma."""

    def __init__(self, reviews: DataFrame, review_themes: DataFrame):
        self.reviews = reviews
        self.review_themes = review_themes

    def analyze(self, filters: AnalyticsFilters) -> Dict[str, Any]:
        selected = filters.validated()
        current_reviews = apply_filters(self.reviews, selected, "reviews").cache()
        comparison = baseline_filters(selected)
        baseline_reviews = (
            apply_filters(self.reviews, comparison, "reviews").cache()
            if comparison
            else None
        )
        try:
            current_population = _population_summary(current_reviews)
            current_counts = _theme_counts(current_reviews, self.review_themes)
            baseline_population = (
                _population_summary(baseline_reviews)
                if baseline_reviews is not None
                else None
            )
            baseline_counts = (
                _theme_counts(baseline_reviews, self.review_themes)
                if baseline_reviews is not None
                else {}
            )
        finally:
            current_reviews.unpersist()
            if baseline_reviews is not None:
                baseline_reviews.unpersist()

        versions = [
            row.classifier_version
            for row in self.review_themes.select("classifier_version")
            .distinct()
            .collect()
        ]
        if len(versions) != 1:
            raise ValueError("review_themes must contain one classifier version")

        themes = []
        for theme in ALL_THEMES:
            current_mentions = current_counts.get(theme, 0)
            baseline_mentions = baseline_counts.get(theme, 0) if comparison else None
            current_rate = _safe_rate(
                current_mentions, current_population["negative_text_reviews"]
            )
            baseline_rate = (
                _safe_rate(
                    baseline_mentions,
                    baseline_population["negative_text_reviews"],
                )
                if baseline_population is not None
                else None
            )
            change = (
                current_rate - baseline_rate
                if current_rate is not None and baseline_rate is not None
                else None
            )
            themes.append(
                {
                    "theme": theme,
                    "current_mentions": current_mentions,
                    "current_mention_rate": current_rate,
                    "baseline_mentions": baseline_mentions,
                    "baseline_mention_rate": baseline_rate,
                    "mention_rate_change": change,
                    "is_explanatory_candidate": theme not in FALLBACK_THEMES,
                }
            )

        ranked = [
            item
            for item in themes
            if item["is_explanatory_candidate"] and item["current_mentions"] > 0
        ]
        ranked.sort(
            key=lambda item: (
                -(
                    item["mention_rate_change"]
                    if item["mention_rate_change"] is not None
                    else item["current_mention_rate"] or 0.0
                ),
                -(item["current_mention_rate"] or 0.0),
                -item["current_mentions"],
                item["theme"],
            )
        )
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank

        return {
            "schema_version": "1.0",
            "theme_metric_version": THEME_METRIC_VERSION,
            "classifier_version": versions[0],
            "filters": asdict(selected),
            "baseline_filters": asdict(comparison) if comparison else None,
            "population_scope": "negative reviews with eligible text",
            "mention_rate_denominator": "negative_text_reviews",
            "multi_label_note": "Mention rates do not sum to one because themes are multi-label.",
            "ranking_note": (
                "With a baseline, themes are ranked by mention-rate change and then "
                "current support; the ranking is not a causal probability."
            ),
            "current_population": current_population,
            "baseline_population": baseline_population,
            "themes": themes,
            "ranked_hypotheses": ranked,
        }
