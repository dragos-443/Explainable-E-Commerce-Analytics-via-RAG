"""Export small, reproducible CSV samples from curated Parquet datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, MapType, StructType

from ecommerce_rag.common.config import load_config


SUPPORTED_DATASETS = {
    "orders_enriched": "random_seeded",
    "reviews_enriched": "stratified_review_score_and_text_shape",
    "review_order_links": "random_seeded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SUPPORTED_DATASETS), required=True)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--environment", default="local")
    parser.add_argument("--output-root", default="/workspace/data/exports")
    args = parser.parse_args()
    if args.rows < 1 or args.rows > 10_000:
        parser.error("--rows must be between 1 and 10000")
    return args


def random_sample(frame: DataFrame, rows: int, seed: int) -> DataFrame:
    """Return a reproducible pseudo-random sample without replacement."""
    return frame.orderBy(F.rand(seed)).limit(rows)


def review_stratum() -> F.Column:
    text_shape = (
        F.when(F.col("has_title") & F.col("has_message"), "title_and_message")
        .when(F.col("has_title"), "title_only")
        .when(F.col("has_message"), "message_only")
        .otherwise("without_text")
    )
    return F.concat_ws("|", F.col("review_score").cast("string"), text_shape)


def stratified_review_sample(frame: DataFrame, rows: int, seed: int) -> DataFrame:
    """Balance eligible reviews across score and available-text shape."""
    candidates = frame.where(F.col("text_is_eligible")).withColumn(
        "_sample_stratum", review_stratum()
    )
    window = Window.partitionBy("_sample_stratum").orderBy(
        F.rand(seed), F.col("review_id")
    )
    return (
        candidates.withColumn("_within_stratum", F.row_number().over(window))
        .orderBy("_within_stratum", "_sample_stratum")
        .limit(rows)
        .drop("_within_stratum", "_sample_stratum")
    )


def csv_compatible(frame: DataFrame) -> DataFrame:
    """Encode nested Spark values as JSON so every column fits in CSV."""
    expressions = []
    for field in frame.schema.fields:
        column = F.col(field.name)
        if isinstance(field.dataType, (ArrayType, MapType, StructType)):
            column = F.to_json(column)
        expressions.append(column.alias(field.name))
    return frame.select(*expressions)


def write_csv(frame: DataFrame, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = frame.collect()
    with destination.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=frame.columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.asDict(recursive=True))
    return len(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.environment)
    spark = SparkSession.builder.appName("phase-1-export-sample").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    try:
        source_uri = (
            f"{config['storage']['curated_uri']}/olist/{args.dataset}"
        )
        source = spark.read.parquet(source_uri)
        source_rows = source.count()

        if args.dataset == "reviews_enriched":
            candidate_rows = source.where(F.col("text_is_eligible")).count()
            selected = stratified_review_sample(source, args.rows, args.seed)
        else:
            candidate_rows = source_rows
            selected = random_sample(source, args.rows, args.seed)

        selected = csv_compatible(selected)
        output_directory = Path(args.output_root) / args.dataset
        exported_rows = write_csv(selected, output_directory / "sample.csv")
        metadata = {
            "dataset": args.dataset,
            "source": source_uri,
            "source_rows": source_rows,
            "candidate_rows": candidate_rows,
            "rows_requested": args.rows,
            "rows_exported": exported_rows,
            "sampling": SUPPORTED_DATASETS[args.dataset],
            "seed": args.seed,
            "encoding": "utf-8-sig",
        }
        with (output_directory / "metadata.json").open(
            "w", encoding="utf-8"
        ) as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
            metadata_file.write("\n")

        print("EXPORT_SUMMARY=" + json.dumps(metadata, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
