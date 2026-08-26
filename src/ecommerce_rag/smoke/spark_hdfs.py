"""Read text from HDFS with Spark and write a Parquet result back to HDFS."""

from __future__ import annotations

import argparse
import json

from pyspark.sql import SparkSession, functions as functions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = SparkSession.builder.appName("phase-0-hdfs-smoke").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        source = spark.read.text(args.input_uri)
        transformed = (
            source.select(
                functions.upper(functions.trim("value")).alias("normalized_review")
            )
            .where(functions.length("normalized_review") > 0)
            .repartition(4)
        )
        row_count = transformed.count()
        transformed.write.mode("overwrite").parquet(args.output_uri)

        verification_count = spark.read.parquet(args.output_uri).count()
        if row_count == 0 or verification_count != row_count:
            raise RuntimeError(
                f"Parquet verification failed: written={row_count}, "
                f"read={verification_count}"
            )

        print(
            "SMOKE_SPARK_RESULT="
            + json.dumps(
                {
                    "input_uri": args.input_uri,
                    "output_uri": args.output_uri,
                    "rows": verification_count,
                },
                sort_keys=True,
            )
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
