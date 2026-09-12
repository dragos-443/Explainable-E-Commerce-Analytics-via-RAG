from ecommerce_rag.evaluation.embedding_comparison import chunks, percentile


def test_chunks_preserves_all_values() -> None:
    assert list(chunks(["a", "b", "c", "d", "e"], 2)) == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_percentile_uses_nearest_observed_value() -> None:
    assert percentile([3.0, 1.0, 4.0, 2.0], 0.95) == 4.0
    assert percentile([], 0.95) == 0.0
