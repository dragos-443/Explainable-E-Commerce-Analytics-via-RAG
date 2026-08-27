"""Independent checks of Phase 1 artifacts persisted in HDFS."""

from __future__ import annotations

import pytest
from pyspark.sql import DataFrame, SparkSession, functions as F

from ecommerce_rag.common.config import load_config


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("phase-1-output-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    yield session
    session.stop()


def _read(spark: SparkSession, layer: str, name: str) -> DataFrame:
    storage = load_config("local")["storage"]
    return spark.read.parquet(f"{storage[f'{layer}_uri']}/olist/{name}")


def _duplicate_keys(frame: DataFrame, key: list[str]) -> int:
    return frame.groupBy(*key).count().where(F.col("count") > 1).count()


def test_curated_outputs_preserve_declared_grains(spark: SparkSession) -> None:
    orders = _read(spark, "processed", "orders")
    orders_enriched = _read(spark, "curated", "orders_enriched")
    reviews_canonical = _read(spark, "processed", "reviews_canonical")
    reviews_enriched = _read(spark, "curated", "reviews_enriched")

    assert orders.count() == orders_enriched.count()
    assert _duplicate_keys(orders_enriched, ["order_id"]) == 0
    assert reviews_canonical.count() == reviews_enriched.count()
    assert _duplicate_keys(reviews_enriched, ["review_id"]) == 0


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("items_by_order", ["order_id"]),
        ("payments_by_order", ["order_id"]),
        ("geolocation_by_zip", ["geolocation_zip_code_prefix"]),
        ("review_order_links", ["review_id", "order_id"]),
    ],
)
def test_processed_aggregations_have_unique_keys(
    spark: SparkSession, name: str, key: list[str]
) -> None:
    assert _duplicate_keys(_read(spark, "processed", name), key) == 0


def test_review_bridge_references_existing_entities(spark: SparkSession) -> None:
    links = _read(spark, "curated", "review_order_links")
    reviews = _read(spark, "curated", "reviews_enriched").select("review_id")
    orders = _read(spark, "curated", "orders_enriched").select("order_id")

    assert links.join(reviews, "review_id", "left_anti").count() == 0
    assert links.join(orders, "order_id", "left_anti").count() == 0
