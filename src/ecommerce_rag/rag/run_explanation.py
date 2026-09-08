"""CLI for the Phase 5 Analytics + RAG grounded explanation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import SparkSession

from ecommerce_rag.analytics.engine import AnalyticsEngine
from ecommerce_rag.analytics.theme_evidence import ThemeEvidenceEngine
from ecommerce_rag.common.config import load_config
from ecommerce_rag.rag.explanation_pipeline import ExplanationPipeline
from ecommerce_rag.rag.prompting.local_llm import build_generator
from ecommerce_rag.rag.question_interpreter import (
    SUPPORTED_INTENTS,
    SUPPORTED_REQUESTED_THEMES,
    interpret_question,
)
from ecommerce_rag.rag.retrieval.factory import build_review_retriever
from ecommerce_rag.rag.translation import CachedMarianTranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--environment", default="local")
    parser.add_argument("--query-id", default="grounded-explanation")
    parser.add_argument("--intent", choices=SUPPORTED_INTENTS)
    parser.add_argument("--metric")
    parser.add_argument("--requested-theme", choices=SUPPORTED_REQUESTED_THEMES)
    parser.add_argument("--product-category")
    parser.add_argument("--customer-state")
    parser.add_argument("--start-month")
    parser.add_argument("--end-month")
    parser.add_argument("--themes-limit", type=int, default=3)
    parser.add_argument("--evidence-per-theme", type=int, default=2)
    parser.add_argument("--no-translate", action="store_true")
    parser.add_argument("--output-root", default="/workspace/reports/integration/phase5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.themes_limit < 1 or args.evidence_per_theme < 1:
        raise ValueError("themes-limit and evidence-per-theme must be positive")
    config = load_config(args.environment)
    question = interpret_question(
        args.question,
        intent=args.intent,
        metric=args.metric,
        requested_theme=args.requested_theme,
        category=args.product_category,
        customer_state=args.customer_state,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    spark = (
        SparkSession.builder.appName("phase-5-grounded-explanation")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        curated = "{}/olist".format(config["storage"]["curated_uri"])
        orders = spark.read.parquet("{}/orders_enriched".format(curated)).cache()
        reviews = spark.read.parquet("{}/reviews_enriched".format(curated)).cache()
        review_themes = spark.read.parquet("{}/review_themes".format(curated)).cache()
        rag = config["rag"]
        translate = not args.no_translate
        translator = (
            CachedMarianTranslator(
                rag["translation_model"],
                rag["translation_revision"],
                rag["translation_cache"],
                target_prefix=rag["translation_target_prefix"],
            )
            if translate
            else None
        )
        llm = config["llm"]
        generator = build_generator(llm)
        result = ExplanationPipeline(
            AnalyticsEngine(orders, reviews),
            ThemeEvidenceEngine(reviews, review_themes),
            build_review_retriever(config, translator),
            generator,
        ).run(
            question,
            query_id=args.query_id,
            themes_limit=args.themes_limit,
            evidence_per_theme=args.evidence_per_theme,
            translate=translate,
        )
        result["llm"] = {
            "provider": result["llm_backend"]["provider"],
            "model": result["llm_backend"]["model"],
            "revision": llm.get("revision"),
        }
        output = Path(args.output_root) / "{}.json".format(args.query_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "PHASE5_RESULT="
            + json.dumps(
                {
                    "query_id": result["query_id"],
                    "intent": result["question"]["intent"],
                    "insufficient_evidence": result["insufficient_evidence"],
                    "generation_status": (
                        result["generation"].get("generation_status")
                        if result["generation"]
                        else None
                    ),
                    "output": str(output),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        print(result["answer_it"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
