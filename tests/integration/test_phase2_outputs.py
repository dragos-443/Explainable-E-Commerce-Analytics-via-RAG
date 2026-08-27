"""Validate Phase 2 EDA artifacts persisted in HDFS and locally."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pyspark.sql import SparkSession, functions as F


HDFS_EDA = "hdfs://namenode:9000/data/outputs/eda/phase2"
LOCAL_EDA = Path("/workspace/reports/eda/phase2")


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("phase-2-output-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_rating_and_monthly_kpis_are_reconciled(spark: SparkSession) -> None:
    rating = spark.read.parquet(f"{HDFS_EDA}/rating_distribution")
    monthly = spark.read.parquet(f"{HDFS_EDA}/monthly_review_statistics")

    rating_totals = rating.agg(
        F.sum("review_count").alias("reviews"), F.sum("share").alias("share")
    ).first()
    assert rating_totals.reviews == 98_410
    assert rating_totals.share == pytest.approx(1.0)
    assert rating.count() == 5

    assert monthly.where(
        "eligible_for_temporal_comparison and review_count < 500"
    ).count() == 0
    assert monthly.where("rating_variation is not null").count() > 0


def test_selected_delivery_and_category_signals_exist(spark: SparkSession) -> None:
    delivery = {
        row.review_score: row
        for row in spark.read.parquet(f"{HDFS_EDA}/delivery_by_rating").collect()
    }
    categories = spark.read.parquet(f"{HDFS_EDA}/category_statistics")

    assert delivery[1].late_delivery_rate > delivery[5].late_delivery_rate
    office = categories.where("product_category = 'office_furniture'").first()
    assert office.review_count >= 500
    assert office.eligible_for_comparison


def test_complaint_theme_denominator_and_keys_are_consistent(
    spark: SparkSession,
) -> None:
    prevalence = spark.read.parquet(f"{HDFS_EDA}/complaint_theme_prevalence")
    assignments = spark.read.parquet(f"{HDFS_EDA}/complaint_theme_assignments")

    assert prevalence.select("denominator_negative_text_reviews").distinct().first()[0] == 10_861
    assert {row.complaint_theme for row in prevalence.collect()} >= {
        "other",
        "uncertain",
        "non_delivery",
        "delivery_delay",
    }
    duplicate_pairs = (
        assignments.groupBy("review_id", "complaint_theme")
        .count()
        .where("count > 1")
        .count()
    )
    assert duplicate_pairs == 0

    by_category = spark.read.parquet(f"{HDFS_EDA}/complaint_themes_by_category")
    by_month = spark.read.parquet(f"{HDFS_EDA}/complaint_themes_by_month")
    assert by_category.where("product_category = 'office_furniture'").count() > 0
    assert by_month.where("purchase_year = 2018 and purchase_month = 3").count() > 0


def test_local_report_contains_findings_tables_and_figures() -> None:
    findings = json.loads((LOCAL_EDA / "candidate_findings.json").read_text(encoding="utf-8"))
    assert len(findings) == 3
    assert all(item["interpretation_limit"] for item in findings)

    expected_figures = {
        "rating_distribution.png",
        "monthly_rating_trend.png",
        "late_delivery_by_rating.png",
        "negative_rate_by_category.png",
        "complaint_theme_prevalence.png",
        "monthly_order_volume.png",
    }
    assert expected_figures == {
        path.name for path in (LOCAL_EDA / "figures").glob("*.png")
    }
    assert all(path.stat().st_size > 10_000 for path in (LOCAL_EDA / "figures").glob("*.png"))
