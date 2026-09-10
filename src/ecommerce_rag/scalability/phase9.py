"""Generate coherent scaled datasets and benchmark representative Spark workloads."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ecommerce_rag.common.config import load_config


DATASETS = (
    "orders_enriched",
    "reviews_enriched",
    "review_order_links",
    "review_themes",
)
DEFAULT_FACTORS = (1, 5, 10, 25, 50, 100)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="local")
    parser.add_argument("--mode", choices=("generate", "benchmark", "all"), default="all")
    parser.add_argument("--factors", nargs="+", type=positive_int, default=DEFAULT_FACTORS)
    parser.add_argument("--repetitions", type=positive_int, default=3)
    parser.add_argument("--warmup-runs", type=positive_int, default=1)
    parser.add_argument(
        "--output-root", default="/workspace/reports/scalability/phase9"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_factors(factors: Sequence[int]) -> List[int]:
    normalized = sorted(set(int(factor) for factor in factors))
    if not normalized or any(factor <= 0 for factor in normalized):
        raise ValueError("scaling factors must be positive")
    return normalized


def _json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hadoop_path(spark: SparkSession, uri: str):
    path = spark._jvm.org.apache.hadoop.fs.Path(uri)
    filesystem = path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())
    return filesystem, path


def path_is_complete(spark: SparkSession, uri: str) -> bool:
    filesystem, path = _hadoop_path(spark, f"{uri}/_SUCCESS")
    return bool(filesystem.exists(path))


def path_size_bytes(spark: SparkSession, uri: str) -> int:
    filesystem, path = _hadoop_path(spark, uri)
    return int(filesystem.getContentSummary(path).getLength())


def _suffix(column, replica):
    return F.when(
        column.isNull(), column
    ).otherwise(F.concat(column.cast("string"), F.lit("__p9_"), replica.cast("string")))


def scaled_frame(frame: DataFrame, dataset: str, factor: int) -> DataFrame:
    """Duplicate a curated frame while keeping keys aligned across datasets."""
    replicas = F.broadcast(frame.sparkSession.range(factor).select(F.col("id").alias("replica")))
    scaled = frame.crossJoin(replicas)
    if dataset == "orders_enriched":
        for column in ("order_id", "customer_id", "customer_unique_id"):
            if column in scaled.columns:
                scaled = scaled.withColumn(column, _suffix(F.col(column), F.col("replica")))
    elif dataset == "reviews_enriched":
        scaled = scaled.withColumn(
            "review_id", _suffix(F.col("review_id"), F.col("replica"))
        )
        if "linked_order_ids" in scaled.columns:
            scaled = scaled.withColumn(
                "linked_order_ids",
                F.transform(
                    "linked_order_ids",
                    lambda order_id: _suffix(order_id, F.col("replica")),
                ),
            )
    elif dataset == "review_order_links":
        scaled = scaled.withColumn(
            "review_id", _suffix(F.col("review_id"), F.col("replica"))
        ).withColumn("order_id", _suffix(F.col("order_id"), F.col("replica")))
    elif dataset == "review_themes":
        scaled = scaled.withColumn(
            "review_id", _suffix(F.col("review_id"), F.col("replica"))
        )
    else:
        raise ValueError(f"unsupported scaling dataset: {dataset}")
    target_partitions = min(max(frame.rdd.getNumPartitions() * factor, 4), 64)
    return scaled.drop("replica").repartition(target_partitions)


def _duplicate_key_groups(frame: DataFrame, keys: Sequence[str]) -> int:
    return frame.groupBy(*keys).count().where(F.col("count") > 1).count()


def validate_scaled_relationships(
    spark: SparkSession, scaled_root: str, factor: int
) -> Dict[str, Any]:
    root = f"{scaled_root}/phase9/{factor}x"
    orders = spark.read.parquet(f"{root}/orders_enriched").select("order_id")
    reviews = spark.read.parquet(f"{root}/reviews_enriched").select("review_id")
    links = spark.read.parquet(f"{root}/review_order_links").select(
        "review_id", "order_id"
    )
    themes = spark.read.parquet(f"{root}/review_themes")
    checks = {
        "duplicate_order_ids": _duplicate_key_groups(orders, ["order_id"]),
        "duplicate_review_ids": _duplicate_key_groups(reviews, ["review_id"]),
        "duplicate_review_order_links": _duplicate_key_groups(
            links, ["review_id", "order_id"]
        ),
        "duplicate_review_theme_rows": _duplicate_key_groups(
            themes,
            [
                column
                for column in ("review_id", "theme", "classifier_version")
                if column in themes.columns
            ],
        ),
        "orphan_link_order_ids": links.select("order_id")
        .distinct()
        .join(orders, "order_id", "left_anti")
        .count(),
        "orphan_link_review_ids": links.select("review_id")
        .distinct()
        .join(reviews, "review_id", "left_anti")
        .count(),
        "orphan_theme_review_ids": themes.select("review_id")
        .distinct()
        .join(reviews, "review_id", "left_anti")
        .count(),
    }
    passed = all(value == 0 for value in checks.values())
    if not passed:
        raise ValueError(f"scaled relationship validation failed at {factor}x: {checks}")
    return {"factor": factor, "passed": True, "checks": checks}


def generate_scaled_datasets(
    spark: SparkSession,
    curated_root: str,
    scaled_root: str,
    factors: Sequence[int],
    output_root: Path,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    base_frames = {
        name: spark.read.parquet(f"{curated_root}/{name}") for name in DATASETS
    }
    base_counts = {name: frame.count() for name, frame in base_frames.items()}
    generated = []
    relationship_validations = []
    for factor in factors:
        for dataset, frame in base_frames.items():
            target = f"{scaled_root}/phase9/{factor}x/{dataset}"
            expected_rows = base_counts[dataset] * factor
            operation = "generated"
            elapsed = 0.0
            actual_rows = None
            if not force and path_is_complete(spark, target):
                actual_rows = spark.read.parquet(target).count()
                if actual_rows == expected_rows:
                    operation = "reused"
            if operation == "generated":
                started = time.perf_counter()
                scaled_frame(frame, dataset, factor).write.mode("overwrite").parquet(target)
                elapsed = time.perf_counter() - started
                actual_rows = spark.read.parquet(target).count()
            if actual_rows != expected_rows:
                raise ValueError(
                    f"row-count mismatch for {dataset} {factor}x: "
                    f"expected {expected_rows}, found {actual_rows}"
                )
            item = {
                "factor": factor,
                "dataset": dataset,
                "uri": target,
                "operation": operation,
                "base_rows": base_counts[dataset],
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "generation_seconds": elapsed,
                "size_bytes": path_size_bytes(spark, target),
                "schema_matches_base": (
                    spark.read.parquet(target).schema == frame.schema
                ),
            }
            if not item["schema_matches_base"]:
                raise ValueError(f"schema mismatch for {dataset} {factor}x")
            generated.append(item)
            print("PHASE9_DATASET=" + json.dumps(item, sort_keys=True))
        validation = validate_scaled_relationships(spark, scaled_root, factor)
        relationship_validations.append(validation)
        print("PHASE9_RELATIONSHIPS=" + json.dumps(validation, sort_keys=True))
    manifest = {
        "schema_version": "1.0",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_excluded_from_benchmark": True,
        "factors": list(factors),
        "datasets": list(DATASETS),
        "base_counts": base_counts,
        "outputs": generated,
        "relationship_validations": relationship_validations,
    }
    _json_write(output_root / "generation_manifest.json", manifest)
    return manifest


def workload_a(frames: Dict[str, DataFrame]) -> Dict[str, Any]:
    grouped = frames["orders_enriched"].groupBy(
        "purchase_year", "purchase_month", "order_category"
    ).agg(
        F.count("*").alias("orders"),
        F.sum("order_value").alias("order_value_sum"),
        F.avg(F.col("is_late").cast("double")).alias("late_delivery_rate"),
    )
    row = grouped.agg(
        F.count("*").alias("result_groups"),
        F.sum("orders").alias("primary_measure"),
        F.sum("order_value_sum").alias("secondary_measure"),
    ).first()
    return row.asDict()


def workload_b(frames: Dict[str, DataFrame]) -> Dict[str, Any]:
    reviews = frames["reviews_enriched"].select("review_id", "review_score")
    orders = frames["orders_enriched"].select(
        "order_id", "order_category", "customer_state"
    )
    joined = frames["review_order_links"].join(reviews, "review_id", "inner").join(
        orders, "order_id", "inner"
    )
    grouped = joined.groupBy("order_category", "customer_state").agg(
        F.count("*").alias("review_order_rows"),
        F.avg("review_score").alias("average_rating"),
        F.sum(F.when(F.col("review_score") <= 2, 1).otherwise(0)).alias(
            "negative_reviews"
        ),
    )
    row = grouped.agg(
        F.count("*").alias("result_groups"),
        F.sum("review_order_rows").alias("primary_measure"),
        F.sum("negative_reviews").alias("secondary_measure"),
    ).first()
    return row.asDict()


def workload_c(frames: Dict[str, DataFrame]) -> Dict[str, Any]:
    reviews = frames["reviews_enriched"].select(
        "review_id", "review_score", "text_is_eligible"
    )
    orders = frames["orders_enriched"].select(
        "order_id",
        "order_category",
        "purchase_year",
        "purchase_month",
        "order_value",
        "delivery_delay_days",
        "is_late",
    )
    themes = frames["review_themes"].select("review_id", "theme")
    joined = (
        frames["review_order_links"]
        .join(reviews, "review_id", "inner")
        .join(orders, "order_id", "inner")
        .join(themes, "review_id", "left")
        .withColumn(
            "delivery_bucket",
            F.when(F.col("is_late"), F.lit("late"))
            .when(F.col("delivery_delay_days").isNull(), F.lit("unknown"))
            .otherwise(F.lit("on_time")),
        )
        .withColumn(
            "negative_text_signal",
            F.when(
                (F.col("review_score") <= 2) & F.col("text_is_eligible"), 1
            ).otherwise(0),
        )
    )
    grouped = joined.groupBy(
        "purchase_year",
        "purchase_month",
        "order_category",
        "theme",
        "delivery_bucket",
    ).agg(
        F.count("*").alias("joined_rows"),
        F.countDistinct("order_id").alias("distinct_orders"),
        F.countDistinct("review_id").alias("distinct_reviews"),
        F.avg("order_value").alias("average_order_value"),
        F.avg("review_score").alias("average_rating"),
        F.sum("negative_text_signal").alias("negative_text_signals"),
    )
    row = grouped.agg(
        F.count("*").alias("result_groups"),
        F.sum("joined_rows").alias("primary_measure"),
        F.sum("negative_text_signals").alias("secondary_measure"),
    ).first()
    return row.asDict()


WORKLOADS: Dict[str, Callable[[Dict[str, DataFrame]], Dict[str, Any]]] = {
    "A_simple_aggregation": workload_a,
    "B_join_aggregation": workload_b,
    "C_multi_join_features_aggregation": workload_c,
}


def logical_input_rows(workload: str, counts: Dict[str, int]) -> int:
    if workload == "A_simple_aggregation":
        return counts["orders_enriched"]
    if workload == "B_join_aggregation":
        return (
            counts["orders_enriched"]
            + counts["reviews_enriched"]
            + counts["review_order_links"]
        )
    return sum(counts[name] for name in DATASETS)


def _safe_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def benchmark(
    spark: SparkSession,
    scaled_root: str,
    factors: Sequence[int],
    repetitions: int,
    warmup_runs: int,
    output_root: Path,
    environment: str = "local",
    storage_scheme: str = "hdfs",
) -> Dict[str, Any]:
    if 1 not in factors:
        raise ValueError("the benchmark factors must include the 1x baseline")
    runs = []
    warmups = []
    factor_metadata = {}
    for factor in factors:
        paths = {
            name: f"{scaled_root}/phase9/{factor}x/{name}" for name in DATASETS
        }
        missing = [uri for uri in paths.values() if not path_is_complete(spark, uri)]
        if missing:
            raise ValueError(f"scaled datasets missing for {factor}x: {missing}")
        counts = {
            name: spark.read.parquet(uri).count() for name, uri in paths.items()
        }
        factor_metadata[str(factor)] = {
            "row_counts": counts,
            "size_bytes": sum(path_size_bytes(spark, uri) for uri in paths.values()),
        }
        for workload_name, workload in WORKLOADS.items():
            for warmup in range(1, warmup_runs + 1):
                spark.catalog.clearCache()
                frames = {name: spark.read.parquet(uri) for name, uri in paths.items()}
                started = time.perf_counter()
                checksum = {
                    key: _safe_number(value) for key, value in workload(frames).items()
                }
                elapsed = time.perf_counter() - started
                warmup_record = {
                    "factor": factor,
                    "workload": workload_name,
                    "warmup": warmup,
                    "execution_seconds": elapsed,
                    "checksum": checksum,
                    "excluded_from_summary": True,
                }
                warmups.append(warmup_record)
                print("PHASE9_WARMUP=" + json.dumps(warmup_record, sort_keys=True))
            for repetition in range(1, repetitions + 1):
                spark.catalog.clearCache()
                frames = {name: spark.read.parquet(uri) for name, uri in paths.items()}
                input_rows = logical_input_rows(workload_name, counts)
                started = time.perf_counter()
                checksum = {
                    key: _safe_number(value) for key, value in workload(frames).items()
                }
                elapsed = time.perf_counter() - started
                run = {
                    "factor": factor,
                    "workload": workload_name,
                    "repetition": repetition,
                    "execution_seconds": elapsed,
                    "logical_input_rows": input_rows,
                    "throughput_rows_per_second": input_rows / elapsed,
                    "checksum": checksum,
                }
                runs.append(run)
                print("PHASE9_RUN=" + json.dumps(run, sort_keys=True))
    summary_rows = add_scaling_indicators(summarize_runs(runs))
    validate_results(runs, summary_rows, factors)
    payload = {
        "schema_version": "1.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "factors": list(factors),
        "repetitions": repetitions,
        "warmup_runs_per_configuration": warmup_runs,
        "workload_definitions": {
            "A_simple_aggregation": "group orders by month and category; aggregate volume, value and late-delivery rate",
            "B_join_aggregation": "join review-order links, reviews and orders; aggregate rating and negative reviews",
            "C_multi_join_features_aggregation": "join links, reviews, orders and themes; derive delivery and negative-text features; aggregate KPIs",
        },
        "factor_metadata": factor_metadata,
        "environment": environment_metadata(spark, environment, storage_scheme),
        "warmups": warmups,
        "runs": runs,
        "summary": summary_rows,
        "validation": {
            "repeat_checksums_identical": True,
            "additive_results_scale_linearly": True,
        },
        "limitations": [
            "CPU and memory were not sampled because container-level sampling would perturb these short jobs.",
            "Runs share the operating-system and HDFS caches; results characterize this local Docker topology only.",
            "Synthetic replication preserves relationships but does not introduce new business distributions.",
        ],
    }
    _json_write(output_root / "benchmark_runs.json", payload)
    _json_write(
        output_root / "summary.json",
        {key: value for key, value in payload.items() if key != "runs"},
    )
    build_plots(summary_rows, output_root)
    return payload


def summarize_runs(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["factor"], run["workload"]), []).append(run)
    summary = []
    for (factor, workload), items in sorted(grouped.items()):
        elapsed = [item["execution_seconds"] for item in items]
        throughput = [item["throughput_rows_per_second"] for item in items]
        summary.append(
            {
                "factor": factor,
                "workload": workload,
                "repetitions": len(items),
                "logical_input_rows": items[0]["logical_input_rows"],
                "mean_execution_seconds": statistics.mean(elapsed),
                "stdev_execution_seconds": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
                "mean_throughput_rows_per_second": statistics.mean(throughput),
                "stdev_throughput_rows_per_second": statistics.stdev(throughput) if len(throughput) > 1 else 0.0,
                "checksum": items[0]["checksum"],
            }
        )
    return summary


def add_scaling_indicators(
    summary: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    baselines = {
        row["workload"]: row
        for row in summary
        if row["factor"] == min(
            candidate["factor"]
            for candidate in summary
            if candidate["workload"] == row["workload"]
        )
    }
    enriched = []
    for row in summary:
        item = dict(row)
        baseline = baselines[row["workload"]]
        item["execution_time_growth_vs_1x"] = (
            row["mean_execution_seconds"] / baseline["mean_execution_seconds"]
        )
        item["throughput_growth_vs_1x"] = (
            row["mean_throughput_rows_per_second"]
            / baseline["mean_throughput_rows_per_second"]
        )
        enriched.append(item)
    return enriched


def validate_results(
    runs: Sequence[Dict[str, Any]],
    summary: Sequence[Dict[str, Any]],
    factors: Sequence[int],
) -> None:
    for row in summary:
        checksums = [
            run["checksum"]
            for run in runs
            if run["factor"] == row["factor"] and run["workload"] == row["workload"]
        ]
        if not checksums or any(checksum != checksums[0] for checksum in checksums[1:]):
            raise ValueError(f"non-deterministic checksum for {row['workload']} {row['factor']}x")
    baseline_factor = min(factors)
    by_workload = {
        workload: {
            row["factor"]: float(row["checksum"]["primary_measure"])
            for row in summary
            if row["workload"] == workload
        }
        for workload in WORKLOADS
    }
    for workload, values in by_workload.items():
        baseline_per_replica = values[baseline_factor] / baseline_factor
        for factor, value in values.items():
            if not math.isclose(
                value / factor, baseline_per_replica, rel_tol=1e-9, abs_tol=1e-9
            ):
                raise ValueError(f"non-linear additive result for {workload} {factor}x")


def environment_metadata(
    spark: SparkSession,
    environment: str = "local",
    storage_scheme: str = "hdfs",
) -> Dict[str, Any]:
    memory_kib = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    memory_kib = int(line.split()[1])
                    break
    except OSError:
        pass
    conf = spark.sparkContext.getConf()
    return {
        "spark_version": spark.version,
        "spark_master": spark.sparkContext.master,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus_visible_to_driver": os.cpu_count(),
        "driver_memory_total_bytes": memory_kib * 1024 if memory_kib else None,
        "executor_cores": conf.get("spark.executor.cores", "cluster default"),
        "executor_memory": conf.get("spark.executor.memory", "cluster default"),
        "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        "environment": environment,
        "storage": (
            "Amazon S3 with Parquet/Snappy"
            if storage_scheme in {"s3", "s3a"}
            else "HDFS with Parquet/Snappy"
        ),
        "cluster_topology": (
            "Amazon EMR on YARN; node counts recorded by the infrastructure manifest"
            if spark.sparkContext.master == "yarn"
            else "1 Spark master + 2 Spark workers; 1 HDFS NameNode + 2 DataNodes"
        ),
    }


def build_plots(summary: Sequence[Dict[str, Any]], output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "A_simple_aggregation": "A - aggregazione",
        "B_join_aggregation": "B - join + aggregazione",
        "C_multi_join_features_aggregation": "C - multi-join + feature",
    }
    for metric, ylabel, filename in (
        ("mean_execution_seconds", "Tempo medio (secondi)", "execution_time_vs_scale.png"),
        (
            "mean_throughput_rows_per_second",
            "Throughput medio (righe logiche/s)",
            "throughput_vs_scale.png",
        ),
    ):
        fig, axis = plt.subplots(figsize=(8, 5))
        for workload in WORKLOADS:
            rows = sorted(
                (row for row in summary if row["workload"] == workload),
                key=lambda row: row["factor"],
            )
            axis.plot(
                [row["factor"] for row in rows],
                [row[metric] for row in rows],
                marker="o",
                label=labels[workload],
            )
        axis.set_xlabel("Fattore di scala")
        axis.set_ylabel(ylabel)
        axis.set_xticks(sorted({row["factor"] for row in summary}))
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_root / filename, dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    factors = normalize_factors(args.factors)
    config = load_config(args.environment)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.appName("phase-9-scalability")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        curated_root = f"{config['storage']['curated_uri']}/olist"
        scaled_root = config["storage"]["scaled_uri"]
        if args.mode in {"generate", "all"}:
            generate_scaled_datasets(
                spark,
                curated_root,
                scaled_root,
                factors,
                output_root,
                force=args.force,
            )
            spark.catalog.clearCache()
        if args.mode in {"benchmark", "all"}:
            benchmark(
                spark,
                scaled_root,
                factors,
                args.repetitions,
                args.warmup_runs,
                output_root,
                environment=args.environment,
                storage_scheme=config["storage"]["scheme"],
            )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
