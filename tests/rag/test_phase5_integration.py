"""Unit tests for the Phase 5 Analytics + RAG integration contracts."""

from __future__ import annotations

from pyspark.sql import SparkSession

from ecommerce_rag.analytics.engine import AnalyticsFilters
from ecommerce_rag.analytics.theme_evidence import ThemeEvidenceEngine
from ecommerce_rag.rag.context_builder.builder import (
    build_grounded_context,
    render_answer_it,
)
from ecommerce_rag.rag.prompting.grounded import validate_generation
from ecommerce_rag.rag.question_interpreter import interpret_question


def test_question_interpreter_extracts_controlled_fields() -> None:
    question = interpret_question(
        "Perché il rating dei mobili per ufficio è diminuito a marzo 2018?"
    )

    assert question.intent == "explain_rating_drop"
    assert question.metric == "average_rating"
    assert question.category == "office_furniture"
    assert question.start_month == "2018-03"
    assert question.end_month == "2018-03"
    assert question.question_original.startswith("Perché")


def test_question_interpreter_marks_order_volume_as_descriptive() -> None:
    question = interpret_question(
        "Perché è cambiato il volume degli ordini a marzo 2018?"
    )
    assert question.intent == "descriptive_only"
    assert question.metric == "order_volume"


def test_theme_rates_use_all_negative_text_reviews_and_baseline() -> None:
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("phase-5-theme-test")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    try:
        reviews = spark.createDataFrame(
            [
                ("b1", "books", "SP", 2017, 12, 1, True),
                ("b2", "books", "SP", 2018, 1, 5, True),
                ("c1", "books", "SP", 2018, 3, 1, True),
                ("c2", "books", "SP", 2018, 3, 2, True),
                ("c3", "books", "SP", 2018, 3, 5, False),
            ],
            "review_id string, product_category string, customer_state string, "
            "purchase_year int, purchase_month int, review_score int, "
            "text_is_eligible boolean",
        )
        themes = spark.createDataFrame(
            [
                ("b1", "delivery_delay", "rules-test"),
                ("b2", "other", "rules-test"),
                ("c1", "non_delivery", "rules-test"),
                ("c2", "delivery_delay", "rules-test"),
                ("c3", "other", "rules-test"),
            ],
            ["review_id", "theme", "classifier_version"],
        )
        result = ThemeEvidenceEngine(reviews, themes).analyze(
            AnalyticsFilters(product_category="books", start_month="2018-03")
        )
    finally:
        spark.stop()

    current = result["current_population"]
    assert current["total_reviews"] == 3
    assert current["negative_reviews"] == 2
    assert current["negative_text_reviews"] == 2
    assert current["negative_text_coverage"] == 1.0
    by_theme = {item["theme"]: item for item in result["themes"]}
    assert by_theme["non_delivery"]["current_mention_rate"] == 0.5
    assert by_theme["non_delivery"]["baseline_mention_rate"] == 0.0
    assert by_theme["delivery_delay"]["mention_rate_change"] == -0.5
    assert result["ranked_hypotheses"][0]["theme"] == "non_delivery"


def test_llm_contract_rejects_unknown_evidence_and_causal_certainty() -> None:
    valid = {
        "interpretation": "Le evidenze sono compatibili con le ipotesi selezionate. Il tema emerge nei commenti citati.",
        "theme_keys": ["non_delivery"],
        "review_ids": ["a" * 32],
        "evidence_limit": "Le evidenze sono descrittive e non dimostrano un rapporto causale né rappresentano tutta la popolazione.",
    }
    assert validate_generation(valid, ["non_delivery"], ["a" * 32]) == valid

    invalid = dict(valid, review_ids=["b" * 32])
    try:
        validate_generation(invalid, ["non_delivery"], ["a" * 32])
        raise AssertionError("unknown review_id was accepted")
    except ValueError:
        pass
    invalid = dict(valid, interpretation="Le evidenze sono compatibili con le ipotesi selezionate perché il tema causa il calo.")
    try:
        validate_generation(invalid, ["non_delivery"], ["a" * 32])
        raise AssertionError("causal claim was accepted")
    except ValueError:
        pass


def test_context_keeps_original_translation_and_spark_observations() -> None:
    question = interpret_question("Quali problemi emergono dalle recensioni negative?")
    analytics = {
        "metrics": {
            "negative_review_rate": {
                "value": 0.2,
                "baseline_value": None,
                "change": None,
                "denominator": 10,
            },
            "average_rating": {"value": 3.5, "baseline_value": None, "change": None, "denominator": 10},
            "late_delivery_rate": {"value": 0.1, "baseline_value": None, "change": None, "denominator": 10},
        }
    }
    themes = {
        "ranked_hypotheses": [
            {
                "theme": "non_delivery",
                "current_mentions": 1,
                "current_mention_rate": 0.5,
                "baseline_mentions": None,
                "baseline_mention_rate": None,
                "mention_rate_change": None,
            }
        ],
        "current_population": {
            "total_reviews": 10,
            "text_reviews": 4,
            "overall_text_coverage": 0.4,
            "negative_reviews": 2,
            "negative_text_reviews": 2,
            "negative_text_coverage": 1.0,
        },
    }
    evidence = [
        {
            "review_id": "a" * 32,
            "document_original": "Comment: Não chegou",
            "translation": {"message": "Non è arrivato", "title": None},
        }
    ]
    context = build_grounded_context(
        question.as_dict(), analytics, themes, evidence
    )
    generation = {
        "interpretation": "Il tema è compatibile con le evidenze.",
        "theme_keys": ["non_delivery"],
        "review_ids": ["a" * 32],
        "evidence_limit": "Gli esempi non provano un rapporto causale.",
    }
    answer = render_answer_it(context, generation)

    assert "Não chegou" in answer
    assert "Non è arrivato" in answer
    assert "[{}]".format("a" * 32) in answer
    assert "intero gruppo" in answer
