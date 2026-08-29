"""Prepare versioned RAG documents and review themes in curated storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession

from ecommerce_rag.common.config import load_config
from ecommerce_rag.rag.documents import build_rag_documents, validate_document_contract
from ecommerce_rag.rag.themes import classify_review_themes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="local")
    parser.add_argument(
        "--summary-output",
        default="/workspace/reports/rag/phase4/preparation_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.environment)
    spark = (
        SparkSession.builder.appName("phase-4-rag-preparation")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        curated = "{}/olist".format(config["storage"]["curated_uri"])
        orders = spark.read.parquet("{}/orders_enriched".format(curated)).cache()
        reviews = spark.read.parquet("{}/reviews_enriched".format(curated)).cache()
        links = spark.read.parquet("{}/review_order_links".format(curated)).cache()
        review_themes = classify_review_themes(
            reviews, config["rag"]["theme_classifier_version"]
        ).cache()
        documents = build_rag_documents(reviews, orders, links, review_themes).cache()
        summary = validate_document_contract(reviews, documents, review_themes)

        review_themes.write.mode("overwrite").parquet(
            "{}/review_themes".format(curated)
        )
        documents.write.mode("overwrite").parquet(
            "{}/rag_documents".format(curated)
        )
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("PHASE4_PREPARATION=" + json.dumps(summary, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
