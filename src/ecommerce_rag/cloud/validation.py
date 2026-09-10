"""Build and compare portable Spark output manifests for Phase 10."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ecommerce_rag.common.config import load_config


DATASETS = (
    ("curated", "orders_enriched"),
    ("curated", "reviews_enriched"),
    ("curated", "review_order_links"),
    ("curated", "review_themes"),
    ("curated", "rag_documents"),
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_manifest(spark: Any, environment: str) -> dict[str, Any]:
    """Collect schemas and row counts from the portable curated layer."""
    config = load_config(environment)
    storage = config["storage"]
    datasets = []
    for layer, name in DATASETS:
        uri = f"{storage[f'{layer}_uri']}/olist/{name}"
        frame = spark.read.parquet(uri)
        datasets.append(
            {
                "layer": layer,
                "dataset": name,
                "row_count": frame.count(),
                "schema": json.loads(frame.schema.json()),
            }
        )
    contracts_uri = f"{storage['outputs_uri']}/contracts/phase1"
    contracts = [
        {
            "layer": row["layer"],
            "dataset": row["dataset"],
            "schema": json.loads(row["schema_json"]),
        }
        for row in spark.read.json(contracts_uri)
        .select("layer", "dataset", "schema_json")
        .orderBy("layer", "dataset")
        .collect()
    ]
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "spark_version": spark.version,
        "storage_scheme": storage["scheme"],
        "datasets": datasets,
        "phase1_contracts": contracts,
    }


def compare_manifests(
    local_manifest: dict[str, Any], cloud_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Compare logical outputs while ignoring storage addresses and timestamps."""
    local_datasets = {
        (item["layer"], item["dataset"]): item
        for item in local_manifest["datasets"]
    }
    cloud_datasets = {
        (item["layer"], item["dataset"]): item
        for item in cloud_manifest["datasets"]
    }
    dataset_keys_match = local_datasets.keys() == cloud_datasets.keys()
    dataset_checks = []
    for key in sorted(local_datasets.keys() | cloud_datasets.keys()):
        local_item = local_datasets.get(key)
        cloud_item = cloud_datasets.get(key)
        dataset_checks.append(
            {
                "layer": key[0],
                "dataset": key[1],
                "present_in_both": local_item is not None and cloud_item is not None,
                "schema_compatible": bool(
                    local_item and cloud_item and local_item["schema"] == cloud_item["schema"]
                ),
                "row_count_equal": bool(
                    local_item
                    and cloud_item
                    and local_item["row_count"] == cloud_item["row_count"]
                ),
                "local_row_count": local_item["row_count"] if local_item else None,
                "cloud_row_count": cloud_item["row_count"] if cloud_item else None,
            }
        )

    def contracts(manifest: dict[str, Any]) -> dict[tuple[str, str], Any]:
        return {
            (item["layer"], item["dataset"]): item["schema"]
            for item in manifest["phase1_contracts"]
        }

    local_contracts = contracts(local_manifest)
    cloud_contracts = contracts(cloud_manifest)
    contract_keys_match = local_contracts.keys() == cloud_contracts.keys()
    contract_schemas_match = contract_keys_match and all(
        local_contracts[key] == cloud_contracts[key] for key in local_contracts
    )
    compatible = (
        dataset_keys_match
        and all(
            item["present_in_both"]
            and item["schema_compatible"]
            and item["row_count_equal"]
            for item in dataset_checks
        )
        and contract_schemas_match
    )
    return {
        "schema_version": "1.0",
        "compared_at_utc": datetime.now(timezone.utc).isoformat(),
        "compatible": compatible,
        "local_spark_version": local_manifest["spark_version"],
        "cloud_spark_version": cloud_manifest["spark_version"],
        "dataset_keys_match": dataset_keys_match,
        "contract_keys_match": contract_keys_match,
        "contract_schemas_match": contract_schemas_match,
        "datasets": dataset_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--environment", required=True)
    manifest.add_argument("--output", required=True)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--local", required=True)
    comparison.add_argument("--cloud", required=True)
    comparison.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "compare":
        local = json.loads(Path(args.local).read_text(encoding="utf-8"))
        cloud = json.loads(Path(args.cloud).read_text(encoding="utf-8"))
        result = compare_manifests(local, cloud)
        _write_json(Path(args.output), result)
        print("PHASE10_COMPATIBILITY=" + json.dumps(result, sort_keys=True))
        if not result["compatible"]:
            raise SystemExit(1)
        return

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("phase-10-portability-manifest").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        result = build_manifest(spark, args.environment)
        _write_json(Path(args.output), result)
        summary = {
            "environment": result["environment"],
            "spark_version": result["spark_version"],
            "datasets": {
                item["dataset"]: item["row_count"] for item in result["datasets"]
            },
            "phase1_contract_count": len(result["phase1_contracts"]),
        }
        print("PHASE10_MANIFEST=" + json.dumps(summary, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
