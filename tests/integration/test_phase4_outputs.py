"""Integration checks for Phase 4 curated data, Chroma and retrieval artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pytest
from pyspark.sql import SparkSession, functions as F

from ecommerce_rag.common.config import load_config


CURATED = "hdfs://namenode:9000/data/curated/olist"
REPORT_ROOT = Path("/workspace/reports/rag/phase4")


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("phase-4-output-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_document_population_and_text_sources_reconcile(spark: SparkSession) -> None:
    documents = spark.read.parquet(f"{CURATED}/rag_documents")
    assert documents.count() == 42_380
    assert documents.select("document_id").distinct().count() == 42_380
    assert {
        row.text_source: row["count"]
        for row in documents.groupBy("text_source").count().collect()
    } == {"title": 1_721, "message": 30_863, "title_and_message": 9_796}
    assert documents.where(
        "text_source = 'title' and document_text like 'Title: %' "
        "and document_text not like '%Comment:%'"
    ).count() == 1_721
    assert documents.where(
        "document_text is null or trim(document_text) = ''"
    ).count() == 0


def test_theme_dataset_is_complete_versioned_and_unique(spark: SparkSession) -> None:
    themes = spark.read.parquet(f"{CURATED}/review_themes")
    documents = spark.read.parquet(f"{CURATED}/rag_documents")
    assert themes.count() == 43_405
    assert themes.groupBy("review_id", "theme").count().where("count > 1").count() == 0
    assert themes.select("review_id").distinct().count() == documents.count()
    assert themes.select("classifier_version").distinct().first()[0] == "rules-pt-v3"
    assert {row.theme for row in themes.select("theme").distinct().collect()} >= {
        "non_delivery", "delivery_delay", "other", "uncertain"
    }
    on_time_themes = {
        row.theme
        for row in themes.where(
            "review_id = '9183634de55bb0f2ffa5a9146b253bbc'"
        ).select("theme").collect()
    }
    assert "delivery_delay" not in on_time_themes
    assert on_time_themes == {"damaged_or_defective", "service_or_refund"}


def test_ambiguous_order_metadata_is_not_forced(spark: SparkSession) -> None:
    documents = spark.read.parquet(f"{CURATED}/rag_documents")
    assert documents.where("is_multi_order_review and order_id is not null").count() == 0
    assert documents.where(
        "not is_single_category_order and product_category is not null"
    ).count() == 0


def test_chroma_population_and_title_only_documents() -> None:
    config = load_config("local")
    client = chromadb.HttpClient(
        host=config["chroma"]["host"], port=config["chroma"]["port"]
    )
    collection = client.get_collection(config["chroma"]["collection"])
    assert collection.count() == 42_380
    title_only = collection.get(
        where={"text_source": "title"}, limit=1, include=["documents", "metadatas"]
    )
    assert title_only["ids"]
    assert title_only["documents"][0].startswith("Title: ")
    assert "Comment:" not in title_only["documents"][0]
    assert title_only["metadatas"][0]["document_id"] == title_only["ids"][0]
    assert title_only["metadatas"][0]["metadata_hash"]
    assert title_only["metadatas"][0]["theme_classifier_version"] == "rules-pt-v3"


def test_idempotent_index_run_skips_every_unchanged_document() -> None:
    summary = json.loads(
        (REPORT_ROOT / "index_idempotent.json").read_text(encoding="utf-8")
    )
    assert summary["documents_after"] == 42_380
    assert summary["documents_upserted"] == 0
    assert summary["documents_metadata_updated"] == 0
    assert summary["documents_skipped_unchanged"] == 42_380
    assert summary["stale_documents_removed"] == 0


def test_cross_language_and_filtered_retrieval_outputs() -> None:
    baseline = json.loads(
        (REPORT_ROOT / "non-delivery.json").read_text(encoding="utf-8")
    )
    filtered = json.loads(
        (REPORT_ROOT / "delay-filtered.json").read_text(encoding="utf-8")
    )
    assert baseline["question_language"] == "it"
    assert baseline["result_count"] == 5
    assert any(
        result["metadata"]["theme_non_delivery"]
        for result in baseline["results"]
    )
    assert filtered["result_count"] > 0
    assert all(
        result["metadata"]["review_score"] <= 2
        and result["metadata"]["theme_delivery_delay"]
        and result["metadata"]["product_category"] == "office_furniture"
        for result in filtered["results"]
    )
    assert all(result["document_original"] for result in filtered["results"])
    assert all(result["distance"] is not None for result in filtered["results"])


def test_translation_output_keeps_original_and_uses_cache() -> None:
    first = json.loads((REPORT_ROOT / "translated.json").read_text(encoding="utf-8"))
    cached = json.loads(
        (REPORT_ROOT / "translated-cached.json").read_text(encoding="utf-8")
    )
    assert first["results"][0]["document_original"]
    assert first["results"][0]["translation"]["automatic"] is True
    assert first["results"][0]["translation"]["status"] == "translated"
    assert cached["results"][0]["translation"]["cache_hit"] is True
