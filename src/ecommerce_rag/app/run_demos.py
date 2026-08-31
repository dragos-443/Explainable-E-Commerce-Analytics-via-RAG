"""Run and verify the real Phase 6 demonstration suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession

from ecommerce_rag.analytics.engine import AnalyticsEngine
from ecommerce_rag.analytics.theme_evidence import ThemeEvidenceEngine
from ecommerce_rag.app.demo_cases import DEMO_CASES, get_demo_case, verify_demo_result
from ecommerce_rag.common.config import load_config
from ecommerce_rag.rag.explanation_pipeline import ExplanationPipeline
from ecommerce_rag.rag.prompting.local_llm import LocalTransformersGenerator
from ecommerce_rag.rag.question_interpreter import interpret_question
from ecommerce_rag.rag.retrieval.factory import build_review_retriever
from ecommerce_rag.rag.translation import CachedMarianTranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="local")
    parser.add_argument("--case-id", choices=[case.case_id for case in DEMO_CASES])
    parser.add_argument("--output-root", default="/workspace/reports/demo/phase6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.environment)
    selected_cases = (
        [get_demo_case(args.case_id)] if args.case_id else list(DEMO_CASES)
    )
    spark = (
        SparkSession.builder.appName("phase-6-demo-suite")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        curated = "{}/olist".format(config["storage"]["curated_uri"])
        orders = spark.read.parquet("{}/orders_enriched".format(curated)).cache()
        reviews = spark.read.parquet("{}/reviews_enriched".format(curated)).cache()
        review_themes = spark.read.parquet("{}/review_themes".format(curated)).cache()
        analytics_engine = AnalyticsEngine(orders, reviews)
        overall_reference = analytics_engine.analyze(query_id="demo-overall-reference")
        rag = config["rag"]
        translator = CachedMarianTranslator(
            rag["translation_model"],
            rag["translation_revision"],
            rag["translation_cache"],
            target_prefix=rag["translation_target_prefix"],
        )
        llm = config["llm"]
        pipeline = ExplanationPipeline(
            analytics_engine,
            ThemeEvidenceEngine(reviews, review_themes),
            build_review_retriever(config, translator),
            LocalTransformersGenerator(
                llm["model"], llm["revision"], int(llm["max_new_tokens"])
            ),
        )
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        summaries = []
        for case in selected_cases:
            question = interpret_question(
                case.question,
                category=case.category,
                start_month=case.start_month,
                end_month=case.end_month,
            )
            result = pipeline.run(question, query_id=case.case_id)
            result["llm"] = {
                "provider": llm["provider"],
                "model": llm["model"],
                "revision": llm["revision"],
            }
            verification = verify_demo_result(case, result, overall_reference)
            result["demo_verification"] = verification
            output = output_root / "{}.json".format(case.case_id)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summaries.append(
                {
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "passed": verification["passed"],
                    "failed_checks": [
                        check["name"]
                        for check in verification["checks"]
                        if not check["passed"]
                    ],
                    "output": str(output),
                }
            )
            print(
                "PHASE6_CASE="
                + json.dumps(summaries[-1], ensure_ascii=False, sort_keys=True)
            )
        summary = {
            "schema_version": "1.0",
            "case_count": len(summaries),
            "passed_count": sum(item["passed"] for item in summaries),
            "all_passed": all(item["passed"] for item in summaries),
            "cases": summaries,
            "overall_reference": {
                "average_rating": overall_reference["metrics"]["average_rating"]["value"],
                "negative_review_rate": overall_reference["metrics"]["negative_review_rate"]["value"],
                "late_delivery_rate": overall_reference["metrics"]["late_delivery_rate"]["value"],
            },
        }
        (output_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("PHASE6_SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
        if not summary["all_passed"]:
            raise ValueError("One or more demo cases failed their verification contract")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
