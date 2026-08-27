"""Unit tests for the Phase 3 analytics engine."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DecimalType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from ecommerce_rag.analytics.engine import (
    AnalyticsEngine,
    AnalyticsFilters,
    baseline_filters,
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[1]")
        .appName("phase-3-engine-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture()
def curated_frames(spark: SparkSession):
    order_schema = StructType(
        [
            StructField("order_id", StringType(), False),
            StructField("order_category", StringType(), True),
            StructField("customer_state", StringType(), True),
            StructField("purchase_date", DateType(), True),
            StructField("order_status", StringType(), True),
            StructField("order_value", DecimalType(12, 2), True),
            StructField("is_late", BooleanType(), True),
            StructField("delivery_delay_days", DoubleType(), True),
        ]
    )
    orders = spark.createDataFrame(
        [
            ("b1", "books", "SP", date(2017, 12, 1), "delivered", Decimal("80"), False, -1.0),
            ("b2", "books", "SP", date(2018, 1, 1), "delivered", Decimal("100"), False, -2.0),
            ("b3", "books", "SP", date(2018, 2, 1), "delivered", Decimal("120"), True, 2.0),
            ("c1", "books", "SP", date(2018, 3, 1), "delivered", Decimal("200"), True, 4.0),
            ("c2", "books", "SP", date(2018, 3, 2), "canceled", Decimal("300"), None, None),
            ("x1", "toys", "RJ", date(2018, 3, 1), "delivered", Decimal("999"), False, -3.0),
        ],
        order_schema,
    )
    review_schema = StructType(
        [
            StructField("review_id", StringType(), False),
            StructField("product_category", StringType(), True),
            StructField("customer_state", StringType(), True),
            StructField("purchase_year", IntegerType(), True),
            StructField("purchase_month", IntegerType(), True),
            StructField("review_score", IntegerType(), True),
        ]
    )
    reviews = spark.createDataFrame(
        [
            ("rb1", "books", "SP", 2017, 12, 5),
            ("rb2", "books", "SP", 2018, 1, 4),
            ("rb3", "books", "SP", 2018, 2, 3),
            ("rc1", "books", "SP", 2018, 3, 1),
            ("rc2", "books", "SP", 2018, 3, 2),
            ("rx1", "toys", "RJ", 2018, 3, 5),
        ],
        review_schema,
    )
    return orders, reviews


def test_filter_validation_and_baseline_window() -> None:
    filters = AnalyticsFilters(
        product_category="books", customer_state="sp", start_month="2018-03"
    ).validated()
    baseline = baseline_filters(filters)

    assert filters.customer_state == "SP"
    assert filters.end_month == "2018-03"
    assert baseline.start_month == "2017-12"
    assert baseline.end_month == "2018-02"
    with pytest.raises(ValueError, match="YYYY-MM"):
        AnalyticsFilters(start_month="March 2018").validated()
    with pytest.raises(ValueError, match="after"):
        AnalyticsFilters(start_month="2018-04", end_month="2018-03").validated()


def test_metrics_filters_and_baseline_are_reconciled(curated_frames) -> None:
    orders, reviews = curated_frames
    result = AnalyticsEngine(orders, reviews).analyze(
        AnalyticsFilters(
            product_category="books", customer_state="sp", start_month="2018-03"
        ),
        query_id="books-sp-march",
    )
    metrics = result["metrics"]

    assert result["populations"]["reviews"] == 2
    assert result["populations"]["orders"] == 2
    assert result["populations"]["delivery_eligible_orders"] == 1
    assert metrics["average_rating"]["value"] == pytest.approx(1.5)
    assert metrics["average_rating"]["baseline_value"] == pytest.approx(4.0)
    assert metrics["rating_variation"]["value"] == pytest.approx(-2.5)
    assert metrics["negative_review_rate"]["value"] == pytest.approx(1.0)
    assert metrics["low_rating_distribution"]["1"]["share"] == pytest.approx(0.5)
    assert metrics["late_delivery_rate"]["value"] == pytest.approx(1.0)
    assert metrics["average_delivery_delay"]["value"] == pytest.approx(4.0)
    assert metrics["order_volume"]["value"] == 2
    assert metrics["average_order_value"]["value"] == pytest.approx(250.0)
    assert metrics["order_status_distribution"]["canceled"]["share"] == pytest.approx(0.5)
    assert result["interpretation_limit"]


def test_unfiltered_query_has_no_artificial_baseline(curated_frames) -> None:
    orders, reviews = curated_frames
    result = AnalyticsEngine(orders, reviews).analyze()

    assert result["baseline"]["filters"] is None
    assert result["metrics"]["rating_variation"]["value"] is None
    assert result["metrics"]["average_rating"]["baseline_value"] is None


def test_empty_filtered_population_returns_null_rates(curated_frames) -> None:
    orders, reviews = curated_frames
    result = AnalyticsEngine(orders, reviews).analyze(
        AnalyticsFilters(customer_state="XX")
    )

    assert result["populations"]["reviews"] == 0
    assert result["populations"]["orders"] == 0
    assert result["metrics"]["average_rating"]["value"] is None
    assert result["metrics"]["negative_review_rate"]["value"] is None
    assert result["metrics"]["late_delivery_rate"]["value"] is None
    assert result["metrics"]["order_status_distribution"] == {}
