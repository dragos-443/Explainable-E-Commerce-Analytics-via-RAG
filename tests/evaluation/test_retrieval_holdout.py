import json
from pathlib import Path

import pytest

from ecommerce_rag.evaluation.retrieval_holdout import (
    confidence_interval,
    ndcg_at_k,
    reciprocal_rank,
)
from ecommerce_rag.evaluation.retrieval_final_holdout import (
    annotation_agreement,
    load_queries as load_final_holdout_queries,
    retrieval_query,
)
from ecommerce_rag.evaluation.retrieval_operational_benchmark import (
    selective_retrieval_query,
)
from ecommerce_rag.evaluation.service_refund_improvement import (
    legacy_retrieval_query,
    service_queries,
)
from ecommerce_rag.rag.explanation_pipeline import theme_retrieval_query


QUERY_PATH = Path(
    "/workspace/src/ecommerce_rag/evaluation/data/retrieval_holdout_queries.json"
)
OPERATIONAL_QUERY_PATH = QUERY_PATH.with_name("retrieval_operational_queries.json")


def test_holdout_has_30_unique_balanced_queries() -> None:
    queries = json.loads(QUERY_PATH.read_text(encoding="utf-8"))
    assert len(queries) == 30
    assert len({item["query_id"] for item in queries}) == 30
    assert len({item["question_it"] for item in queries}) == 30
    counts = {}
    for query in queries:
        assert query["filters"]["max_review_score"] == 2
        assert "theme" not in query["filters"]
        aspect = query["expected_aspects"][0]
        counts[aspect] = counts.get(aspect, 0) + 1
    assert set(counts.values()) == {5}


def test_rank_metrics_reward_earlier_relevant_documents() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    assert reciprocal_rank(ranked, {"b", "d"}) == 0.5
    assert ndcg_at_k(ranked, {"a", "b"}, 5) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, set(), 5) == 0.0


def test_bootstrap_interval_is_deterministic() -> None:
    assert confidence_interval([0.0, 1.0, 1.0], seed=7) == confidence_interval(
        [0.0, 1.0, 1.0], seed=7
    )


def test_service_refund_experiment_uses_all_five_target_queries() -> None:
    queries = service_queries()

    assert len(queries) == 5
    assert all(query["expected_aspects"] == ["service_or_refund"] for query in queries)


def test_strict_service_refund_query_excludes_simple_intentions() -> None:
    question = "Quali problemi emergono?"

    legacy = legacy_retrieval_query(question)
    strict = theme_retrieval_query(question, "service_or_refund")

    assert legacy != strict
    assert "non una semplice intenzione" in strict


def test_operational_benchmark_focuses_only_wrong_or_missing_queries() -> None:
    target = {
        "question_it": "Quali ordini sono incompleti?",
        "expected_aspects": ["wrong_or_missing_item"],
    }
    delay = {
        "question_it": "Quali consegne sono in ritardo?",
        "expected_aspects": ["delivery_delay"],
    }

    assert retrieval_query(target) == target["question_it"]
    assert selective_retrieval_query(target) == theme_retrieval_query(
        target["question_it"], "wrong_or_missing_item"
    )
    assert selective_retrieval_query(delay) == delay["question_it"]


def test_final_holdout_adds_50_unique_balanced_queries() -> None:
    queries = load_final_holdout_queries()
    previous = json.loads(QUERY_PATH.read_text(encoding="utf-8"))
    development = json.loads(
        QUERY_PATH.with_name("retrieval_queries.json").read_text(encoding="utf-8")
    )
    previous_ids = {item["query_id"] for item in previous + development}

    assert len(queries) == 50
    assert len({item["query_id"] for item in queries}) == 50
    assert len({item["question_it"] for item in queries}) == 50
    assert not ({item["query_id"] for item in queries} & previous_ids)
    aspect_counts = {}
    for query in queries:
        assert query["filters"]["max_review_score"] == 2
        assert len(query["expected_aspects"]) == 1
        aspect = query["expected_aspects"][0]
        aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
    assert sorted(aspect_counts.values()) == [8, 8, 8, 8, 9, 9]


def test_final_holdout_crosses_six_categories_with_all_aspects() -> None:
    queries = load_final_holdout_queries()
    matrix_categories = {
        "perfumery",
        "stationery",
        "consoles_games",
        "home_appliances",
        "luggage_accessories",
        "musical_instruments",
    }
    expected_aspects = {item["expected_aspects"][0] for item in queries}

    for category in matrix_categories:
        category_aspects = {
            item["expected_aspects"][0]
            for item in queries
            if item["filters"].get("product_category") == category
        }
        assert category_aspects == expected_aspects


def test_operational_benchmark_has_50_supported_queries() -> None:
    queries = json.loads(OPERATIONAL_QUERY_PATH.read_text(encoding="utf-8"))
    other_queries = []
    for filename in (
        "retrieval_queries.json",
        "retrieval_holdout_queries.json",
        "retrieval_final_holdout_queries.json",
    ):
        other_queries.extend(
            json.loads(
                OPERATIONAL_QUERY_PATH.with_name(filename).read_text(encoding="utf-8")
            )
        )
    other_ids = {item["query_id"] for item in other_queries}

    assert len(queries) == 50
    assert len({item["query_id"] for item in queries}) == 50
    assert len({item["question_it"] for item in queries}) == 50
    assert not ({item["query_id"] for item in queries} & other_ids)

    aspect_counts = {}
    dimension_counts = {"general": 0, "category": 0, "state": 0, "month": 0}
    aspects_by_dimension = {name: set() for name in dimension_counts}
    for query in queries:
        filters = query["filters"]
        assert filters["max_review_score"] == 2
        assert len(query["expected_aspects"]) == 1
        aspect = query["expected_aspects"][0]
        aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1

        optional_dimensions = sum(
            (
                "product_category" in filters,
                "customer_state" in filters,
                "start_month" in filters or "end_month" in filters,
            )
        )
        assert optional_dimensions <= 1
        if "product_category" in filters:
            dimension = "category"
        elif "customer_state" in filters:
            dimension = "state"
        elif "start_month" in filters or "end_month" in filters:
            assert filters["start_month"] == filters["end_month"]
            dimension = "month"
        else:
            dimension = "general"
        dimension_counts[dimension] += 1
        aspects_by_dimension[dimension].add(aspect)

    assert sorted(aspect_counts.values()) == [8, 8, 8, 8, 9, 9]
    assert dimension_counts == {"general": 11, "category": 22, "state": 7, "month": 10}
    assert all(len(aspects) == 6 for aspects in aspects_by_dimension.values())


def test_annotation_agreement_reports_rate_and_kappa() -> None:
    primary = {
        "query_file_sha256": "same",
        "queries": {
            "q": [
                {"review_id": "a", "relevant": True},
                {"review_id": "b", "relevant": False},
                {"review_id": "c", "relevant": True},
                {"review_id": "d", "relevant": False},
            ]
        },
    }
    secondary = {
        "query_file_sha256": "same",
        "queries": {
            "q": [
                {"review_id": "a", "relevant": True},
                {"review_id": "b", "relevant": False},
                {"review_id": "c", "relevant": False},
                {"review_id": "d", "relevant": False},
            ]
        },
    }

    result = annotation_agreement(primary, secondary)

    assert result["judged_pairs"] == 4
    assert result["agreement_rate"] == 0.75
    assert result["cohens_kappa"] == 0.5
