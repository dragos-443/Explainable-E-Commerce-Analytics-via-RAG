"""Command-line entry point for one structured analytics query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession

from ecommerce_rag.analytics.engine import AnalyticsEngine, AnalyticsFilters
from ecommerce_rag.common.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="local")
    parser.add_argument("--query-id", default="analytics-query")
    parser.add_argument("--product-category")
    parser.add_argument("--customer-state")
    parser.add_argument("--start-month")
    parser.add_argument("--end-month")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.environment)
    spark = (
        SparkSession.builder.appName("phase-3-analytics-query")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        curated = "{}/olist".format(config["storage"]["curated_uri"])
        orders = spark.read.parquet("{}/orders_enriched".format(curated)).cache()
        reviews = spark.read.parquet("{}/reviews_enriched".format(curated)).cache()
        result = AnalyticsEngine(orders, reviews).analyze(
            AnalyticsFilters(
                product_category=args.product_category,
                customer_state=args.customer_state,
                start_month=args.start_month,
                end_month=args.end_month,
            ),
            query_id=args.query_id,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("PHASE3_RESULT=" + json.dumps(result, ensure_ascii=False, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
