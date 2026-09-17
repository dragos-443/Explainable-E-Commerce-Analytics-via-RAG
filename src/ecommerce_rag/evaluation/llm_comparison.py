"""Controlled comparison of the final local and cloud generation backends."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pyspark.sql import SparkSession

from ecommerce_rag.analytics.engine import AnalyticsEngine
from ecommerce_rag.analytics.theme_evidence import ThemeEvidenceEngine
from ecommerce_rag.app.demo_cases import DEMO_CASES, verify_demo_result
from ecommerce_rag.common.config import load_config
from ecommerce_rag.rag.explanation_pipeline import ExplanationPipeline
from ecommerce_rag.rag.prompting.local_llm import build_generator
from ecommerce_rag.rag.question_interpreter import interpret_question
from ecommerce_rag.rag.retrieval.factory import build_review_retriever
from ecommerce_rag.rag.translation import build_translator


TERRA_PRICES_USD_PER_MILLION = {
    "input": 2.00,
    "cached_input": 0.20,
    "output": 12.00,
}
PRICE_REFERENCE_DATE = "2026-09-08"
PRICE_SOURCE = "https://developers.openai.com/api/docs/pricing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="local")
    parser.add_argument(
        "--output-root",
        default="/workspace/reports/evaluation/comparisons/llm-backends",
    )
    return parser.parse_args()


def normalized_usage(usage: Dict[str, Any] | None) -> Dict[str, int]:
    usage = usage or {}
    details = usage.get("input_tokens_details") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_input_tokens": int(details.get("cached_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
    }


def estimate_terra_cost_usd(usage: Dict[str, Any] | None) -> float:
    tokens = normalized_usage(usage)
    uncached = max(tokens["input_tokens"] - tokens["cached_input_tokens"], 0)
    return (
        uncached * TERRA_PRICES_USD_PER_MILLION["input"]
        + tokens["cached_input_tokens"]
        * TERRA_PRICES_USD_PER_MILLION["cached_input"]
        + tokens["output_tokens"] * TERRA_PRICES_USD_PER_MILLION["output"]
    ) / 1_000_000


class MeasuredGenerator:
    """Preserve generator metadata while recording only the generation call."""

    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.provider = getattr(delegate, "provider", None)
        self.model_name = getattr(delegate, "model_name", None)
        self.last_provider = None
        self.last_model = None
        self.last_usage = None
        self.last_elapsed_seconds = 0.0

    def generate(self, messages: List[Dict[str, str]]) -> str:
        started = time.perf_counter()
        try:
            return self.delegate.generate(messages)
        finally:
            self.last_elapsed_seconds = time.perf_counter() - started
            self.last_provider = getattr(self.delegate, "last_provider", None) or getattr(
                self.delegate, "provider", None
            )
            self.last_model = getattr(self.delegate, "last_model", None) or getattr(
                self.delegate, "model_name", None
            )
            self.last_usage = getattr(self.delegate, "last_usage", None)

    def unload(self) -> None:
        self.delegate.unload()


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def summarize_run(label: str, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    generated = [item for item in cases if item["generation_status"] is not None]
    usage = {
        key: sum(item["usage"][key] for item in cases)
        for key in ("input_tokens", "cached_input_tokens", "output_tokens")
    }
    return {
        "label": label,
        "case_count": len(cases),
        "verification_passed": sum(item["verification_passed"] for item in cases),
        "llm_generated_validated": sum(
            item["generation_status"] == "llm_generated_validated" for item in cases
        ),
        "validated_fallback": sum(
            item["generation_status"] == "validated_fallback" for item in cases
        ),
        "insufficient_evidence": sum(
            item["generation_status"] is None for item in cases
        ),
        "mean_generation_seconds": _mean(
            item["generation_seconds"] for item in generated
        ),
        "median_generation_seconds": (
            statistics.median(item["generation_seconds"] for item in generated)
            if generated
            else None
        ),
        "mean_end_to_end_seconds": _mean(
            item["end_to_end_seconds"] for item in cases
        ),
        "usage": usage,
        "estimated_api_cost_usd": sum(
            item["estimated_api_cost_usd"] for item in cases
        ),
        "cost_note": (
            "Direct API estimate; local compute and network costs excluded."
            if label == "openai"
            else "No direct API charge; local compute cost is not monetized here."
        ),
    }


def evidence_signature(result: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {
            "review_id": item["review_id"],
            "retrieved_for_theme": item["retrieved_for_theme"],
        }
        for item in result["context"]["review_evidence"]
    ]


def run_backend(
    label: str,
    pipeline: ExplanationPipeline,
    measured: MeasuredGenerator,
    overall_reference: Dict[str, Any],
    output_root: Path,
) -> List[Dict[str, Any]]:
    backend_root = output_root / label
    backend_root.mkdir(parents=True, exist_ok=True)
    records = []
    for case in DEMO_CASES:
        question = interpret_question(case.question)
        started = time.perf_counter()
        result = pipeline.run(question, query_id=f"final-ab-{label}-{case.case_id}")
        elapsed = time.perf_counter() - started
        verification = verify_demo_result(case, result, overall_reference)
        result["demo_verification"] = verification
        output = backend_root / f"{case.case_id}.json"
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        generation = result.get("generation") or {}
        usage = normalized_usage(measured.last_usage if result.get("generation") else None)
        record = {
            "case_id": case.case_id,
            "verification_passed": verification["passed"],
            "generation_status": generation.get("generation_status"),
            "fallback_reason": generation.get("fallback_reason"),
            "effective_backend": result["llm_backend"],
            "generation_seconds": measured.last_elapsed_seconds if generation else 0.0,
            "end_to_end_seconds": elapsed,
            "usage": usage,
            "estimated_api_cost_usd": (
                estimate_terra_cost_usd(measured.last_usage)
                if result.get("generation")
                and result["llm_backend"].get("provider") == "openai"
                else 0.0
            ),
            "evidence_signature": evidence_signature(result),
            "interpretation": generation.get("interpretation"),
            "output": str(output),
        }
        records.append(record)
        print("LLM_COMPARISON_CASE=" + json.dumps(record, ensure_ascii=False))
    return records


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required for the controlled comparison")
    config = load_config(args.environment)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.appName("phase-7-final-llm-comparison")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    original_provider = os.environ.get("LLM_PROVIDER")
    try:
        curated = f"{config['storage']['curated_uri']}/olist"
        orders = spark.read.parquet(f"{curated}/orders_enriched").cache()
        reviews = spark.read.parquet(f"{curated}/reviews_enriched").cache()
        review_themes = spark.read.parquet(f"{curated}/review_themes").cache()
        analytics = AnalyticsEngine(orders, reviews)
        overall_reference = analytics.analyze(query_id="final-ab-overall-reference")
        translator = build_translator(config)
        retriever = build_review_retriever(config, translator)
        theme_engine = ThemeEvidenceEngine(reviews, review_themes)
        runs = {}
        for label, provider in (("ollama", "ollama"), ("openai", "openai")):
            os.environ["LLM_PROVIDER"] = provider
            measured = MeasuredGenerator(build_generator(config["llm"]))
            pipeline = ExplanationPipeline(analytics, theme_engine, retriever, measured)
            runs[label] = run_backend(
                label, pipeline, measured, overall_reference, output_root
            )
        paired = []
        for local, cloud in zip(runs["ollama"], runs["openai"]):
            paired.append(
                {
                    "case_id": local["case_id"],
                    "identical_retrieval_evidence": (
                        local["evidence_signature"] == cloud["evidence_signature"]
                    ),
                    "ollama_interpretation": local["interpretation"],
                    "openai_interpretation": cloud["interpretation"],
                }
            )
        comparison = {
            "schema_version": "1.0",
            "protocol": {
                "cases": [case.case_id for case in DEMO_CASES],
                "controlled_variable": "generation backend",
                "question_source": "natural-language question without manual filters",
                "translation_provider": os.getenv("TRANSLATION_PROVIDER", "local"),
                "price_reference_date": PRICE_REFERENCE_DATE,
                "price_source": PRICE_SOURCE,
                "terra_prices_usd_per_million_tokens": TERRA_PRICES_USD_PER_MILLION,
                "historical_artifacts_preserved": True,
            },
            "summaries": {
                label: summarize_run(label, records)
                for label, records in runs.items()
            },
            "paired_cases": paired,
            "all_evidence_identical": all(
                item["identical_retrieval_evidence"] for item in paired
            ),
            "runs": runs,
        }
        comparison_path = output_root / "comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("LLM_COMPARISON_SUMMARY=" + json.dumps(comparison["summaries"]))
        if not comparison["all_evidence_identical"]:
            raise ValueError("The comparison did not use identical retrieval evidence")
    finally:
        if original_provider is None:
            os.environ.pop("LLM_PROVIDER", None)
        else:
            os.environ["LLM_PROVIDER"] = original_provider
        spark.stop()


if __name__ == "__main__":
    main()
