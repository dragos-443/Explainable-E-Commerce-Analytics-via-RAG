from ecommerce_rag.cloud.reporting import comparison_rows


def test_comparison_keeps_only_common_configurations() -> None:
    local = [
        {
            "factor": 1,
            "workload": "A",
            "mean_execution_seconds": 4.0,
            "mean_throughput_rows_per_second": 25.0,
        },
        {
            "factor": 10,
            "workload": "A",
            "mean_execution_seconds": 8.0,
            "mean_throughput_rows_per_second": 125.0,
        },
    ]
    cloud = [
        {
            "factor": 10,
            "workload": "A",
            "mean_execution_seconds": 2.0,
            "mean_throughput_rows_per_second": 500.0,
        }
    ]
    rows = comparison_rows(local, cloud)
    assert len(rows) == 1
    assert rows[0]["cloud_speedup_vs_local"] == 4.0
