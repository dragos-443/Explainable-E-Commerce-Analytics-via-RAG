"""Prepare and score a manually annotated complaint-theme sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

from ecommerce_rag.common.config import load_config
from ecommerce_rag.evaluation.metrics import multilabel_metrics
from ecommerce_rag.rag.themes import ALL_THEMES


DATA_ROOT = Path(__file__).with_name("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "score"))
    parser.add_argument("--environment", default="local")
    parser.add_argument("--per-theme", type=int, default=6)
    parser.add_argument("--output-root", default="/workspace/reports/evaluation/phase7")
    parser.add_argument(
        "--sample-source",
        choices=("reference", "latest"),
        default="reference",
        help=(
            "Score the frozen manually annotated sample (reference) or the most "
            "recently collected annotation template (latest)."
        ),
    )
    return parser.parse_args()


def collect(environment: str, per_theme: int, output_root: Path) -> None:
    config = load_config(environment)
    spark = (
        SparkSession.builder.appName("phase-7-theme-annotation-sample")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        curated = "{}/olist".format(config["storage"]["curated_uri"])
        reviews = spark.read.parquet("{}/reviews_enriched".format(curated))
        themes = spark.read.parquet("{}/review_themes".format(curated))
        predicted = themes.groupBy("review_id").agg(
            F.sort_array(F.collect_set("theme")).alias("predicted_themes")
        )
        candidates = (
            reviews.join(predicted, "review_id")
            .withColumn(
                "text_source",
                F.when(F.col("has_title") & F.col("has_message"), F.lit("title_and_message"))
                .when(F.col("has_title"), F.lit("title"))
                .otherwise(F.lit("message")),
            )
            .select(
                "review_id",
                "review_score",
                "review_text",
                "product_category",
                "purchase_month",
                "text_source",
                "predicted_themes",
            )
            .withColumn(
                "sample_hash", F.sha2(F.concat(F.lit("phase7-42:"), F.col("review_id")), 256)
            )
        )
        selected = []
        selected_ids = set()
        for theme in ALL_THEMES:
            rows = (
                candidates.where(F.array_contains("predicted_themes", theme))
                .orderBy("sample_hash")
                .limit(per_theme * 8)
                .collect()
            )
            rank = 0
            for row in rows:
                if row.review_id in selected_ids:
                    continue
                rank += 1
                selected_ids.add(row.review_id)
                selected.append(
                    {
                        "review_id": row.review_id,
                        "review_score": row.review_score,
                        "review_text_original": row.review_text,
                        "product_category": row.product_category,
                        "purchase_month": row.purchase_month,
                        "text_source": row.text_source,
                        "predicted_themes": row.predicted_themes,
                        "sampling_stratum": theme,
                        "stratum_rank": rank,
                        "pilot": rank <= 3,
                        "manual_themes": None,
                        "annotation_note": "",
                    }
                )
                if rank == per_theme:
                    break
            if rank != per_theme:
                raise ValueError("Not enough distinct examples for {}".format(theme))
        # The balanced three-per-theme pilot has 24 rows; add one prespecified row.
        next(item for item in selected if item["sampling_stratum"] == ALL_THEMES[0] and item["stratum_rank"] == 4)["pilot"] = True
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "theme_annotations_template.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "PHASE7_THEME_SAMPLE="
            + json.dumps(
                {
                    "final_size": len(selected),
                    "pilot_size": sum(item["pilot"] for item in selected),
                    "per_theme": per_theme,
                },
                sort_keys=True,
            )
        )
    finally:
        spark.stop()


def score(output_root: Path, sample_source: str = "reference") -> None:
    template_path = (
        DATA_ROOT / "theme_benchmark_sample.json"
        if sample_source == "reference"
        else output_root / "theme_annotations_template.json"
    )
    template = json.loads(
        template_path.read_text(encoding="utf-8")
    )
    annotations = json.loads(
        (DATA_ROOT / "theme_annotations.json").read_text(encoding="utf-8")
    )
    expected = {key: set(value) for key, value in annotations["manual_themes"].items()}
    predicted = {item["review_id"]: set(item["predicted_themes"]) for item in template}
    if set(expected) != set(predicted):
        raise ValueError("Manual annotations do not exactly cover the selected sample")
    pilot_ids = {item["review_id"] for item in template if item["pilot"]}
    pilot = multilabel_metrics(
        {key: expected[key] for key in pilot_ids},
        {key: predicted[key] for key in pilot_ids},
    )
    final = multilabel_metrics(expected, predicted)
    errors = [
        {
            "review_id": key,
            "expected": sorted(expected[key]),
            "predicted": sorted(predicted[key]),
            "false_positive": sorted(predicted[key] - expected[key]),
            "false_negative": sorted(expected[key] - predicted[key]),
        }
        for key in sorted(expected)
        if expected[key] != predicted[key]
    ]
    composition = {
        "sample_size": len(template),
        "pilot_size": len(pilot_ids),
        "scores": {
            str(score): sum(item["review_score"] == score for item in template)
            for score in range(1, 6)
        },
        "text_sources": {
            source: sum(item["text_source"] == source for item in template)
            for source in sorted({item["text_source"] for item in template})
        },
        "categories": len({item["product_category"] for item in template}),
        "months": len({item["purchase_month"] for item in template}),
        "sampling_per_predicted_theme": 6,
    }
    result = {
        "schema_version": "1.0",
        "annotation_protocol": annotations["annotation_protocol"],
        "sample_selection": annotations["sample_selection"],
        "sample_source": sample_source,
        "composition": composition,
        "pilot_metrics": pilot,
        "final_metrics": final,
        "errors": errors,
    }
    (output_root / "theme_classification_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PHASE7_THEME_METRICS="
        + json.dumps(
            {
                "sample_size": composition["sample_size"],
                "micro_f1": final["micro_f1"],
                "macro_f1": final["macro_f1"],
                "exact_match_ratio": final["exact_match_ratio"],
                "error_count": len(errors),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    if args.per_theme < 3:
        raise ValueError("per-theme must be at least three to form the pilot")
    output_root = Path(args.output_root)
    if args.mode == "collect":
        collect(args.environment, args.per_theme, output_root)
    else:
        score(output_root, args.sample_source)


if __name__ == "__main__":
    main()
