"""Build the long-lived local resources used by the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyspark.sql import SparkSession

from ecommerce_rag.analytics.engine import AnalyticsEngine
from ecommerce_rag.analytics.theme_evidence import ThemeEvidenceEngine
from ecommerce_rag.common.config import load_config
from ecommerce_rag.rag.explanation_pipeline import ExplanationPipeline
from ecommerce_rag.rag.prompting.local_llm import build_generator
from ecommerce_rag.rag.retrieval.factory import build_review_retriever
from ecommerce_rag.rag.translation import build_translator


@dataclass
class ApplicationRuntime:
    spark: Any
    pipeline: ExplanationPipeline


def build_application_runtime(environment: str = "local") -> ApplicationRuntime:
    config = load_config(environment)
    spark = (
        SparkSession.builder.appName("phase-8-streamlit")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    curated = "{}/olist".format(config["storage"]["curated_uri"])
    orders = spark.read.parquet("{}/orders_enriched".format(curated)).cache()
    reviews = spark.read.parquet("{}/reviews_enriched".format(curated)).cache()
    review_themes = spark.read.parquet("{}/review_themes".format(curated)).cache()
    translator = build_translator(config)
    llm = config["llm"]
    pipeline = ExplanationPipeline(
        AnalyticsEngine(orders, reviews),
        ThemeEvidenceEngine(reviews, review_themes),
        build_review_retriever(config, translator),
        build_generator(llm),
    )
    return ApplicationRuntime(spark=spark, pipeline=pipeline)
