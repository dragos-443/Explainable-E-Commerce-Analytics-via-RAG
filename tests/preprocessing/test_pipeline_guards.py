"""Tests for Phase 1 preprocessing invariants."""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

from ecommerce_rag.preprocessing.export_sample import (
    random_sample,
    stratified_review_sample,
)
from ecommerce_rag.preprocessing.pipeline import (
    QualityRecorder,
    assert_unique,
    join_without_fanout,
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[1]")
        .appName("phase-1-guard-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_join_without_fanout_accepts_unique_right_key(spark: SparkSession) -> None:
    left = spark.createDataFrame([("order-1",), ("order-2",)], ["order_id"])
    right = spark.createDataFrame(
        [("order-1", "paid"), ("order-2", "shipped")],
        ["order_id", "status"],
    )

    result = join_without_fanout(
        left, right, "order_id", "unique_join", QualityRecorder()
    )

    assert result.count() == 2


def test_join_without_fanout_rejects_duplicate_right_key(spark: SparkSession) -> None:
    left = spark.createDataFrame([("order-1",), ("order-2",)], ["order_id"])
    right = spark.createDataFrame(
        [("order-1", "item-1"), ("order-1", "item-2")],
        ["order_id", "item_id"],
    )

    with pytest.raises(ValueError, match="Fan-out detected"):
        join_without_fanout(
            left, right, "order_id", "duplicated_join", QualityRecorder()
        )


def test_single_column_logical_key_rejects_null(spark: SparkSession) -> None:
    frame = spark.createDataFrame([("order-1",), (None,)], "order_id string")

    with pytest.raises(ValueError, match="rows with null keys"):
        assert_unique(frame, "orders", ["order_id"], QualityRecorder())


def test_random_export_sample_is_reproducible(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(index,) for index in range(100)], ["id"])

    first = [row.id for row in random_sample(frame, 10, 42).collect()]
    second = [row.id for row in random_sample(frame, 10, 42).collect()]

    assert first == second
    assert len(first) == len(set(first)) == 10


def test_review_export_balances_score_and_text_shape(spark: SparkSession) -> None:
    rows = []
    text_shapes = [(True, False), (False, True), (True, True)]
    for score in range(1, 6):
        for has_title, has_message in text_shapes:
            for repetition in range(2):
                rows.append(
                    (
                        f"{score}-{has_title}-{has_message}-{repetition}",
                        score,
                        has_title,
                        has_message,
                        True,
                    )
                )
    frame = spark.createDataFrame(
        rows,
        [
            "review_id",
            "review_score",
            "has_title",
            "has_message",
            "text_is_eligible",
        ],
    )

    sample = stratified_review_sample(frame, 15, 42)
    represented_strata = sample.select(
        "review_score", "has_title", "has_message"
    ).distinct()

    assert sample.count() == 15
    assert represented_strata.count() == 15
