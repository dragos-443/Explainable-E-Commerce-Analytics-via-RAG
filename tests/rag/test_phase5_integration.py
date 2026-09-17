"""Unit tests for the Phase 5 Analytics + RAG integration contracts."""

from __future__ import annotations

import json

import pytest
from pyspark.sql import SparkSession

from ecommerce_rag.analytics.engine import AnalyticsFilters
from ecommerce_rag.analytics.theme_evidence import ThemeEvidenceEngine
from ecommerce_rag.rag.context_builder.builder import (
    build_grounded_context,
    render_answer_it,
)
from ecommerce_rag.rag.explanation_pipeline import (
    _safe_fallback,
    focus_theme_evidence,
    theme_retrieval_query,
)
from ecommerce_rag.rag.prompting.grounded import (
    assemble_interpretation,
    build_messages,
    validate_generation,
)
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


@pytest.mark.parametrize(
    ("text", "category", "theme"),
    [
        ("Quali problemi di qualità emergono per letto, bagno e tavola?", "bed_bath_table", "quality_or_expectation"),
        ("Quali difetti vengono segnalati per l'elettronica?", "electronics", "damaged_or_defective"),
        ("Quali articoli mancanti riguardano i piccoli elettrodomestici?", "small_appliances", "wrong_or_missing_item"),
        ("Quali problemi di rimborso emergono per i mobili per ufficio?", "office_furniture", "service_or_refund"),
    ],
)
def test_question_interpreter_extracts_requested_theme(
    text: str, category: str, theme: str
) -> None:
    question = interpret_question(text)

    assert question.intent == "analyze_low_rating_complaints"
    assert question.category == category
    assert question.requested_theme == theme
    assert question.interpretation_method == "controlled_rules_v2"


def test_explicit_theme_focus_keeps_only_the_requested_spark_hypothesis() -> None:
    question = interpret_question("Quali difetti emergono nelle recensioni negative?")
    evidence = {
        "ranked_hypotheses": [
            {"theme": "non_delivery", "current_mentions": 20},
            {"theme": "damaged_or_defective", "current_mentions": 8},
        ]
    }

    focused = focus_theme_evidence(question, evidence)

    assert [item["theme"] for item in focused["ranked_hypotheses"]] == [
        "damaged_or_defective"
    ]
    assert focused["requested_theme"] == "damaged_or_defective"
    assert len(evidence["ranked_hypotheses"]) == 2


def test_theme_retrieval_query_focuses_known_themes_only() -> None:
    question = "Quali problemi emergono dalle recensioni negative?"

    focused = theme_retrieval_query(question, "service_or_refund")

    assert focused.startswith(question)
    assert "assistenza già contattata ma senza risposta" in focused
    assert "rimborso già richiesti ma rimasti senza esito" in focused
    assert "non una semplice intenzione" in focused
    assert theme_retrieval_query(question, "other") == question


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
                ("b1", "books", "SP", 2017, 12, 1, "Entrega atrasada", True),
                ("b2", "books", "SP", 2018, 1, 5, "Produto excelente", True),
                ("c1", "books", "SP", 2018, 3, 1, "Pedido não chegou", True),
                ("c2", "books", "SP", 2018, 3, 2, "Entrega atrasada", True),
                ("c3", "books", "SP", 2018, 3, 5, None, False),
                ("c4", "books", "SP", 2018, 3, 1, ".", True),
            ],
            "review_id string, product_category string, customer_state string, "
            "purchase_year int, purchase_month int, review_score int, "
            "review_text string, text_is_eligible boolean",
        )
        themes = spark.createDataFrame(
            [
                ("b1", "delivery_delay", "rules-test"),
                ("b2", "other", "rules-test"),
                ("c1", "non_delivery", "rules-test"),
                ("c2", "delivery_delay", "rules-test"),
                ("c3", "other", "rules-test"),
                ("c4", "uncertain", "rules-test"),
            ],
            ["review_id", "theme", "classifier_version"],
        )
        result = ThemeEvidenceEngine(reviews, themes).analyze(
            AnalyticsFilters(product_category="books", start_month="2018-03")
        )
    finally:
        spark.stop()

    current = result["current_population"]
    assert current["total_reviews"] == 4
    assert current["negative_reviews"] == 3
    assert current["negative_text_reviews"] == 2
    assert current["negative_text_coverage"] == 2 / 3
    by_theme = {item["theme"]: item for item in result["themes"]}
    assert by_theme["non_delivery"]["current_mention_rate"] == 0.5
    assert by_theme["non_delivery"]["baseline_mention_rate"] == 0.0
    assert by_theme["delivery_delay"]["mention_rate_change"] == -0.5
    assert result["ranked_hypotheses"][0]["theme"] == "non_delivery"


def test_llm_contract_rejects_unknown_evidence_and_causal_certainty() -> None:
    review_id = "a" * 32
    valid = {
        "interpretation": (
            "Le recensioni citate descrivono clienti che attendono un ordine mai "
            "arrivato. Questo argomento ricorre negli esempi associati al tema "
            "della mancata consegna e offre una lettura prudente del fenomeno."
        ),
        "review_ids": [review_id],
    }
    validated = validate_generation(
        valid,
        ["non_delivery"],
        [review_id],
        {review_id: "non_delivery"},
        {review_id: "Il cliente afferma che l'ordine non è mai arrivato"},
    )
    assert validated["interpretation"] == valid["interpretation"]
    assert validated["theme_keys"] == ["non_delivery"]
    assert validated["review_ids"] == [review_id]

    invalid = dict(valid, review_ids=["b" * 32])
    try:
        validate_generation(
            invalid,
            ["non_delivery"],
            [review_id],
            {review_id: "non_delivery"},
            {review_id: "Il cliente afferma che l'ordine non è mai arrivato"},
        )
        raise AssertionError("unknown review_id was accepted")
    except ValueError:
        pass
    invalid = dict(
        valid,
        interpretation=(
            "Le recensioni descrivono il problema perché il ritardo causa il calo. "
            "I clienti riportano quindi una relazione certa tra i due fenomeni."
        ),
    )
    try:
        validate_generation(
            invalid,
            ["non_delivery"],
            [review_id],
            {review_id: "non_delivery"},
            {review_id: "Il cliente afferma che l'ordine non è mai arrivato"},
        )
        raise AssertionError("causal claim was accepted")
    except ValueError:
        pass


def test_assembled_interpretation_always_identifies_retrieved_reviews() -> None:
    result = assemble_interpretation(
        {"review_summary": "Prodotto difettoso e non funzionante"},
        ["a" * 32],
        "analyze_low_rating_complaints",
    )

    assert result["interpretation"].startswith(
        "Le recensioni recuperate descrivono quanto segue: prodotto"
    )


def test_llm_contract_rejects_unsupported_concept_despite_lexical_overlap() -> None:
    review_id = "a" * 32
    payload = {
        "interpretation": (
            "Le recensioni recuperate descrivono assistenza senza risposta e un "
            "ordine mai ricevuto. Nel gruppo analizzato queste segnalazioni "
            "descrivono il contesto qualitativo delle recensioni negative."
        ),
        "review_ids": [review_id],
    }

    with pytest.raises(ValueError, match="concept absent"):
        validate_generation(
            payload,
            ["service_or_refund"],
            [review_id],
            {review_id: "service_or_refund"},
            {review_id: "L'assistenza ha ignorato la richiesta del cliente"},
        )


def test_llm_contract_accepts_grounded_synonyms_and_rejects_new_concepts() -> None:
    review_id = "a" * 32
    grounded = {
        "interpretation": (
            "I clienti segnalano una mancata consegna e un ritardo nella consegna. "
            "Questi problemi offrono una possibile chiave di lettura del fenomeno "
            "osservato, senza stabilire un rapporto certo."
        ),
        "review_ids": [review_id],
    }
    source = "Il termine è scaduto e il prodotto non è ancora arrivato"

    validated = validate_generation(
        grounded,
        ["non_delivery"],
        [review_id],
        {review_id: "non_delivery"},
        {review_id: source},
    )

    assert validated["review_ids"] == [review_id]
    unsupported = dict(
        grounded,
        interpretation=(
            "I clienti segnalano esclusivamente un rimborso mai ricevuto. "
            "Questa esperienza offre una possibile chiave di lettura del fenomeno "
            "osservato, senza stabilire un rapporto certo."
        ),
    )
    with pytest.raises(ValueError, match="concept absent"):
        validate_generation(
            unsupported,
            ["non_delivery"],
            [review_id],
            {review_id: "non_delivery"},
            {review_id: source},
        )


def test_llm_contract_recognizes_one_of_two_items_as_incomplete_order() -> None:
    review_id = "a" * 32
    payload = {
        "interpretation": (
            "Le recensioni recuperate descrivono una consegna incompleta e "
            "assistenza senza risposta. Un cliente riferisce di avere ricevuto "
            "soltanto parte delle sedie acquistate e di essere ancora in attesa."
        ),
        "review_ids": [review_id],
    }
    source = (
        "Ho ricevuto una delle due sedie che ho acquistato. "
        "Sono in attesa di una risposta, ma finora niente."
    )

    validated = validate_generation(
        payload,
        ["wrong_or_missing_item"],
        [review_id],
        {review_id: "wrong_or_missing_item"},
        {review_id: source},
    )

    assert validated["review_ids"] == [review_id]


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
            "retrieved_for_theme": "non_delivery",
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
    assert "20.00%" in context["quantitative_summary_it"]
    assert context["quantitative_summary_it"].count("Nel gruppo selezionato") == 1
    assert ", il rating medio è " in context["quantitative_summary_it"]
    assert " e la quota di consegne tardive è " in context["quantitative_summary_it"]
    assert "mancata consegna" in context["theme_summary_it"]
    assert "Limite interpretativo" not in answer

    prompt_payload = json.loads(build_messages(context)[1]["content"])
    prompt_review = prompt_payload["review_evidence"][0]
    assert prompt_review["theme_hypothesis"] == "non_delivery"
    assert prompt_review["original_portuguese"] == "Comment: Não chegou"
    assert prompt_review["automatic_translation_it"] == "Non è arrivato"
    assert "argument_it" not in prompt_review


def test_deterministic_fallback_contains_metrics_themes_and_review_arguments() -> None:
    question = interpret_question("Quali problemi emergono dalle recensioni negative?")
    analytics = {
        "metrics": {
            "negative_review_rate": {
                "value": 0.2,
                "baseline_value": 0.1,
                "change": 0.1,
                "denominator": 10,
            },
            "average_rating": {
                "value": 3.5,
                "baseline_value": 4.0,
                "change": -0.5,
                "denominator": 10,
            },
            "late_delivery_rate": {
                "value": 0.15,
                "baseline_value": 0.05,
                "change": 0.1,
                "denominator": 10,
            },
        }
    }
    themes = {
        "ranked_hypotheses": [
            {
                "theme": "non_delivery",
                "current_mentions": 2,
                "current_mention_rate": 0.5,
                "baseline_mentions": 0,
                "baseline_mention_rate": 0.0,
                "mention_rate_change": 0.5,
            }
        ],
        "current_population": {
            "total_reviews": 10,
            "text_reviews": 4,
            "overall_text_coverage": 0.4,
            "negative_reviews": 4,
            "negative_text_reviews": 4,
            "negative_text_coverage": 1.0,
        },
    }
    evidence = [
        {
            "review_id": "a" * 32,
            "document_original": "Comment: Ainda não recebi o pedido",
            "retrieved_for_theme": "non_delivery",
            "translation": {
                "message": "Non ho ancora ricevuto l'ordine",
                "title": None,
            },
        }
    ]
    context = build_grounded_context(question.as_dict(), analytics, themes, evidence)

    generation = _safe_fallback(context, "invalid LLM output")

    assert "20.00%" in generation["quantitative_summary"]
    assert "mancata consegna" in generation["theme_summary"]
    assert generation["review_arguments"][0]["review_id"] == "a" * 32
    assert "Non ho ancora ricevuto" in generation["review_arguments"][0]["argument_it"]
    assert "Non ho ancora ricevuto" in generation["interpretation"]
    assert "contesto qualitativo" not in generation["interpretation"]
