import pytest

from ecommerce_rag.evaluation.metrics import multilabel_metrics, retrieval_metrics


def test_retrieval_metrics_use_fixed_k_and_judged_pool():
    result = retrieval_metrics(["a", "b", "c"], {"a", "c", "d"}, 3)
    assert result["precision_at_k"] == pytest.approx(2 / 3)
    assert result["recall_at_k"] == pytest.approx(2 / 3)
    assert result["hit_rate_at_k"] == 1.0


def test_multilabel_metrics_report_micro_macro_and_exact_match():
    expected = {"1": {"a", "b"}, "2": {"b"}}
    predicted = {"1": {"a"}, "2": {"b", "c"}}
    result = multilabel_metrics(expected, predicted)
    assert result["micro_precision"] == pytest.approx(2 / 3)
    assert result["micro_recall"] == pytest.approx(2 / 3)
    assert result["micro_f1"] == pytest.approx(2 / 3)
    assert result["exact_match_ratio"] == 0.0
    assert result["per_label"]["a"]["f1"] == 1.0
    assert result["per_label"]["c"]["support"] == 0
