import json
from pathlib import Path

import pytest

from ecommerce_rag.evaluation.retrieval_holdout import (
    confidence_interval,
    ndcg_at_k,
    reciprocal_rank,
)
from ecommerce_rag.evaluation.service_refund_improvement import (
    legacy_retrieval_query,
    service_queries,
)
from ecommerce_rag.rag.explanation_pipeline import theme_retrieval_query


QUERY_PATH = Path(
    "/workspace/src/ecommerce_rag/evaluation/data/retrieval_holdout_queries.json"
)


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
