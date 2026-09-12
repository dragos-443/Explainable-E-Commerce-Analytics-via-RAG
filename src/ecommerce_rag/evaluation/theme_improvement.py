"""Create a blind holdout and compare complaint-theme classifier versions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

from pyspark.sql import SparkSession, functions as F

from ecommerce_rag.common.config import load_config
from ecommerce_rag.evaluation.metrics import multilabel_metrics
from ecommerce_rag.rag.themes import ALL_THEMES, classify_review_themes


DATA_ROOT = Path(__file__).with_name("data")
SEED = "phase7-v3-holdout"
BASELINE_VERSION = "rules-pt-v2"
PROPOSED_VERSION = "rules-pt-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "score", "score-development"))
    parser.add_argument("--environment", default="local")
    parser.add_argument("--per-theme", type=int, default=6)
    parser.add_argument("--output-root", default="/workspace/reports/evaluation/phase7")
    return parser.parse_args()


def _spark(name: str) -> SparkSession:
    spark = (
        SparkSession.builder.appName(name)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def collect(environment: str, per_theme: int, output_root: Path) -> None:
    config = load_config(environment)
    prior = json.loads(
        (DATA_ROOT / "theme_annotations.json").read_text(encoding="utf-8")
    )
    excluded_ids = sorted(prior["manual_themes"])
    spark = _spark("phase-7-theme-v3-holdout")
    try:
        curated = "{}/olist".format(config["storage"]["curated_uri"])
        reviews = spark.read.parquet("{}/reviews_enriched".format(curated))
        predictions = classify_review_themes(reviews, PROPOSED_VERSION).groupBy(
            "review_id"
        ).agg(F.sort_array(F.collect_set("theme")).alias("predicted_themes"))
        candidates = (
            reviews.join(predictions, "review_id")
            .where(~F.col("review_id").isin(excluded_ids))
            .withColumn(
                "text_source",
                F.when(
                    F.col("has_title") & F.col("has_message"),
                    F.lit("title_and_message"),
                )
                .when(F.col("has_title"), F.lit("title"))
                .otherwise(F.lit("message")),
            )
            .withColumn(
                "sample_hash",
                F.sha2(F.concat(F.lit(SEED + ":"), F.col("review_id")), 256),
            )
            .select(
                "review_id",
                "review_score",
                "review_text",
                "product_category",
                "purchase_month",
                "text_source",
                "predicted_themes",
                "sample_hash",
            )
        )
        selected = []
        selected_ids: Set[str] = set()
        for theme in ALL_THEMES:
            rows = (
                candidates.where(F.array_contains("predicted_themes", theme))
                .orderBy("sample_hash")
                .limit(per_theme * 8)
                .collect()
            )
            accepted = 0
            for row in rows:
                if row.review_id in selected_ids:
                    continue
                selected_ids.add(row.review_id)
                accepted += 1
                # Predictions and sampling strata are deliberately omitted so
                # manual judgments can be made without seeing model output.
                selected.append(
                    {
                        "review_id": row.review_id,
                        "review_score": row.review_score,
                        "review_text_original": row.review_text,
                        "product_category": row.product_category,
                        "purchase_month": row.purchase_month,
                        "text_source": row.text_source,
                        "manual_themes": None,
                        "annotation_note": "",
                    }
                )
                if accepted == per_theme:
                    break
            if accepted != per_theme:
                raise ValueError("Not enough distinct holdout examples for {}".format(theme))
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "theme_v3_holdout_annotations_template.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "PHASE7_THEME_V3_HOLDOUT="
            + json.dumps(
                {
                    "sample_size": len(selected),
                    "per_predicted_stratum": per_theme,
                    "prior_ids_excluded": len(excluded_ids),
                    "predictions_hidden": True,
                    "seed": SEED,
                },
                sort_keys=True,
            )
        )
    finally:
        spark.stop()


def _predictions(
    spark: SparkSession, template: Iterable[dict], classifier_version: str
) -> Dict[str, Set[str]]:
    rows = [
        (
            item["review_id"],
            int(item["review_score"]),
            item["review_text_original"],
            True,
        )
        for item in template
    ]
    frame = spark.createDataFrame(
        rows,
        "review_id string, review_score int, review_text string, text_is_eligible boolean",
    )
    predictions: Dict[str, Set[str]] = {}
    for row in classify_review_themes(frame, classifier_version).collect():
        predictions.setdefault(row.review_id, set()).add(row.theme)
    return predictions


def _errors(
    expected: Dict[str, Set[str]], predicted: Dict[str, Set[str]]
) -> List[dict]:
    return [
        {
            "review_id": review_id,
            "expected": sorted(expected[review_id]),
            "predicted": sorted(predicted[review_id]),
            "false_positive": sorted(predicted[review_id] - expected[review_id]),
            "false_negative": sorted(expected[review_id] - predicted[review_id]),
        }
        for review_id in sorted(expected)
        if expected[review_id] != predicted[review_id]
    ]


def _comparison(template: List[dict], annotations: dict) -> dict:
    expected = {
        review_id: set(themes)
        for review_id, themes in annotations["manual_themes"].items()
    }
    template_ids = {item["review_id"] for item in template}
    if set(expected) != template_ids:
        raise ValueError("Manual annotations do not exactly cover the V3 holdout")
    spark = _spark("phase-7-theme-v3-score")
    try:
        baseline = _predictions(spark, template, BASELINE_VERSION)
        proposed = _predictions(spark, template, PROPOSED_VERSION)
    finally:
        spark.stop()
    return {
        "schema_version": "1.0",
        "annotation_protocol": annotations["annotation_protocol"],
        "sample_selection": annotations["sample_selection"],
        "sample_size": len(template),
        "baseline": {
            "classifier_version": BASELINE_VERSION,
            "metrics": multilabel_metrics(expected, baseline),
            "errors": _errors(expected, baseline),
        },
        "proposed": {
            "classifier_version": PROPOSED_VERSION,
            "metrics": multilabel_metrics(expected, proposed),
            "errors": _errors(expected, proposed),
        },
    }


def _print_metrics(result: dict) -> None:
    print(
        "PHASE7_THEME_V3_METRICS="
        + json.dumps(
            {
                version: {
                    "micro_f1": result[key]["metrics"]["micro_f1"],
                    "macro_f1": result[key]["metrics"]["macro_f1"],
                    "exact_match_ratio": result[key]["metrics"]["exact_match_ratio"],
                    "error_count": len(result[key]["errors"]),
                }
                for version, key in (
                    (BASELINE_VERSION, "baseline"),
                    (PROPOSED_VERSION, "proposed"),
                )
            },
            sort_keys=True,
        )
    )


def score(output_root: Path) -> None:
    template = json.loads(
        (output_root / "theme_v3_holdout_annotations_template.json").read_text(
            encoding="utf-8"
        )
    )
    annotations = json.loads(
        (DATA_ROOT / "theme_v3_holdout_annotations.json").read_text(encoding="utf-8")
    )
    result = _comparison(template, annotations)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "theme_classifier_improvement_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_metrics(result)


def score_development(output_root: Path) -> None:
    annotations = json.loads(
        (DATA_ROOT / "theme_annotations.json").read_text(encoding="utf-8")
    )
    frozen = json.loads(
        (DATA_ROOT / "theme_development_predictions.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        review_id: set(themes)
        for review_id, themes in annotations["manual_themes"].items()
    }
    predictions = {
        side: {
            review_id: set(themes)
            for review_id, themes in frozen[side]["predictions"].items()
        }
        for side in ("baseline", "proposed")
    }
    if any(set(values) != set(expected) for values in predictions.values()):
        raise ValueError("Frozen predictions do not cover the development sample")
    result = {
        "schema_version": "1.0",
        "annotation_protocol": annotations["annotation_protocol"],
        "sample_selection": annotations["sample_selection"],
        "sample_size": len(expected),
        **{
            side: {
                "classifier_version": frozen[side]["classifier_version"],
                "metrics": multilabel_metrics(expected, predictions[side]),
                "errors": _errors(expected, predictions[side]),
            }
            for side in ("baseline", "proposed")
        },
    }
    (output_root / "theme_classifier_development_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print_metrics(result)


def main() -> None:
    args = parse_args()
    if args.per_theme < 1:
        raise ValueError("per-theme must be positive")
    output_root = Path(args.output_root)
    if args.mode == "collect":
        collect(args.environment, args.per_theme, output_root)
    elif args.mode == "score-development":
        score_development(output_root)
    else:
        score(output_root)


if __name__ == "__main__":
    main()
