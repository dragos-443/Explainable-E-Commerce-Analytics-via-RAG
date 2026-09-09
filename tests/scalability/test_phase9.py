import pytest

from ecommerce_rag.scalability.phase9 import (
    add_scaling_indicators,
    logical_input_rows,
    normalize_factors,
    summarize_runs,
)


def test_normalize_factors_sorts_and_removes_duplicates() -> None:
    assert normalize_factors([10, 1, 5, 5]) == [1, 5, 10]
    with pytest.raises(ValueError, match="positive"):
        normalize_factors([0, 1])


def test_logical_input_rows_matches_each_frozen_workload() -> None:
    counts = {
        "orders_enriched": 10,
        "reviews_enriched": 8,
        "review_order_links": 9,
        "review_themes": 4,
    }
    assert logical_input_rows("A_simple_aggregation", counts) == 10
    assert logical_input_rows("B_join_aggregation", counts) == 27
    assert logical_input_rows("C_multi_join_features_aggregation", counts) == 31


def test_summarize_runs_computes_mean_and_sample_standard_deviation() -> None:
    runs = [
        {
            "factor": 1,
            "workload": "A_simple_aggregation",
            "execution_seconds": 2.0,
            "logical_input_rows": 100,
            "throughput_rows_per_second": 50.0,
            "checksum": {"primary_measure": 100},
        },
        {
            "factor": 1,
            "workload": "A_simple_aggregation",
            "execution_seconds": 4.0,
            "logical_input_rows": 100,
            "throughput_rows_per_second": 25.0,
            "checksum": {"primary_measure": 100},
        },
    ]

    summary = summarize_runs(runs)[0]

    assert summary["mean_execution_seconds"] == 3.0
    assert summary["stdev_execution_seconds"] == pytest.approx(2**0.5)
    assert summary["mean_throughput_rows_per_second"] == 37.5


def test_scaling_indicators_compare_each_workload_with_its_baseline() -> None:
    rows = [
        {
            "factor": 1,
            "workload": "A",
            "mean_execution_seconds": 2.0,
            "mean_throughput_rows_per_second": 100.0,
        },
        {
            "factor": 5,
            "workload": "A",
            "mean_execution_seconds": 5.0,
            "mean_throughput_rows_per_second": 200.0,
        },
    ]

    scaled = add_scaling_indicators(rows)

    assert scaled[0]["execution_time_growth_vs_1x"] == 1.0
    assert scaled[1]["execution_time_growth_vs_1x"] == 2.5
    assert scaled[1]["throughput_growth_vs_1x"] == 2.0
