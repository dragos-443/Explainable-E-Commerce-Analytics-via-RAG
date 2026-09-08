"""Reproducible offline and online efficiency measurements for Phase 7."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Dict

import pyspark
from pyspark.sql import SparkSession

from ecommerce_rag.analytics.engine import AnalyticsEngine
from ecommerce_rag.analytics.theme_evidence import ThemeEvidenceEngine
from ecommerce_rag.common.config import load_config
from ecommerce_rag.rag.embeddings.model import MultilingualE5Embedder
from ecommerce_rag.rag.explanation_pipeline import ExplanationPipeline
from ecommerce_rag.rag.index import chroma_client, get_collection
from ecommerce_rag.rag.prompting.local_llm import build_generator
from ecommerce_rag.rag.question_interpreter import interpret_question
from ecommerce_rag.rag.retrieval.reranking import MultilingualCrossEncoderReranker
from ecommerce_rag.rag.retrieval.service import ReviewRetriever
from ecommerce_rag.rag.translation import CachedMarianTranslator


TEMP_COLLECTION = "phase7_efficiency_temporary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="local")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--chroma-bytes", type=int, required=True)
    parser.add_argument("--translation-cache-bytes", type=int, required=True)
    parser.add_argument("--output-root", default="/workspace/reports/evaluation/phase7")
    return parser.parse_args()


class Timer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.values: Dict[str, float] = {}

    def add(self, key: str, elapsed: float) -> None:
        self.values[key] = self.values.get(key, 0.0) + elapsed


class TimedEmbedder:
    def __init__(self, delegate, timer: Timer) -> None:
        self.delegate = delegate
        self.timer = timer

    def embed_query(self, text: str):
        started = time.perf_counter()
        result = self.delegate.embed_query(text)
        self.timer.add("query_embedding", time.perf_counter() - started)
        return result

    def unload(self) -> None:
        self.delegate.unload()


class TimedCollection:
    def __init__(self, delegate, timer: Timer) -> None:
        self.delegate = delegate

        self.timer = timer

    def query(self, **kwargs):
        started = time.perf_counter()
        result = self.delegate.query(**kwargs)
        self.timer.add("chroma_retrieval", time.perf_counter() - started)
        return result


class TimedTranslator:
    def __init__(self, delegate, timer: Timer) -> None:
        self.delegate = delegate
        self.timer = timer
        self.cache_hits = 0
        self.cache_misses = 0

    def translate_review(self, *args, **kwargs):
        started = time.perf_counter()
        result = self.delegate.translate_review(*args, **kwargs)
        self.timer.add("translation_top_k", time.perf_counter() - started)
        if result.get("cache_hit"):
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return result

    def unload(self) -> None:
        self.delegate.unload()


class TimedReranker:
    def __init__(self, delegate, timer: Timer) -> None:
        self.delegate = delegate
        self.timer = timer

    def rerank(self, *args, **kwargs):
        started = time.perf_counter()
        result = self.delegate.rerank(*args, **kwargs)
        self.timer.add("reranking", time.perf_counter() - started)
        return result

    def unload(self) -> None:
        self.delegate.unload()


class TimedGenerator:
    def __init__(self, delegate, timer: Timer) -> None:
        self.delegate = delegate
        self.timer = timer

    def generate(self, messages):
        started = time.perf_counter()
        result = self.delegate.generate(messages)
        self.timer.add("llm_generation", time.perf_counter() - started)
        return result

    def unload(self) -> None:
        self.delegate.unload()


class TimedAnalyzer:
    def __init__(self, delegate, timer: Timer, key: str) -> None:
        self.delegate = delegate
        self.timer = timer
        self.key = key

    def analyze(self, *args, **kwargs):
        started = time.perf_counter()
        result = self.delegate.analyze(*args, **kwargs)
        self.timer.add(self.key, time.perf_counter() - started)
        return result


def read_first_line(path: str, prefix: str = "") -> str:
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not prefix or line.startswith(prefix):
                return line.split(":", 1)[-1].strip()
    except OSError:
        pass
    return "unknown"


def hardware() -> dict:
    import torch

    memory_limit = read_first_line("/sys/fs/cgroup/memory.max")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_model": read_first_line("/proc/cpuinfo", "model name"),
        "mem_total": read_first_line("/proc/meminfo", "MemTotal"),
        "container_memory_limit_bytes": memory_limit,
        "torch_threads": torch.get_num_threads(),
        "spark_version": pyspark.__version__,
    }


def summarize(rows, keys):
    result = {}
    for key in keys:
        values = [row[key] for row in rows]
        result[key] = {
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
    return result


def offline_benchmark(config: dict, embedder, sample_size: int) -> dict:
    client = chroma_client(config)
    source = get_collection(client, config)
    sample = source.get(
        limit=sample_size, include=["documents", "metadatas"]
    )
    documents = sample["documents"]
    if len(documents) != sample_size:
        raise ValueError("Requested offline sample is not available")
    started = time.perf_counter()
    embeddings = embedder.embed_documents(documents)
    first_embedding_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warm_embeddings = embedder.embed_documents(documents)
    warm_embedding_seconds = time.perf_counter() - started
    try:
        client.delete_collection(TEMP_COLLECTION)
    except Exception:
        pass
    temporary = client.create_collection(
        TEMP_COLLECTION, metadata={"hnsw:space": "cosine", "temporary": "true"}
    )
    try:
        started = time.perf_counter()
        temporary.upsert(
            ids=["eval-{}".format(value) for value in sample["ids"]],
            documents=documents,
            metadatas=sample["metadatas"],
            embeddings=warm_embeddings,
        )
        upsert_seconds = time.perf_counter() - started
        if temporary.count() != sample_size:
            raise ValueError("Temporary Chroma benchmark count mismatch")
    finally:
        client.delete_collection(TEMP_COLLECTION)
    initial = json.loads(
        Path("/workspace/reports/rag/phase4/index_initial.json").read_text(encoding="utf-8")
    )
    return {
        "full_initial_build": {
            "documents": initial["documents_upserted"],
            "total_seconds": initial["elapsed_seconds"],
            "overall_throughput_documents_per_second": initial["documents_upserted"] / initial["elapsed_seconds"],
            "source": "phase-4 initial measured build",
        },
        "controlled_sample": {
            "documents": sample_size,
            "model_load_plus_first_embedding_seconds": first_embedding_seconds,
            "warm_embedding_seconds": warm_embedding_seconds,
            "warm_embedding_throughput_documents_per_second": sample_size / warm_embedding_seconds,
            "chroma_upsert_seconds": upsert_seconds,
            "chroma_upsert_throughput_documents_per_second": sample_size / upsert_seconds,
            "warm_total_seconds": warm_embedding_seconds + upsert_seconds,
        },
    }


def main() -> None:
    args = parse_args()
    if args.sample_size < 1 or args.repetitions < 2:
        raise ValueError("sample-size must be positive and repetitions at least two")
    config = load_config(args.environment)
    rag = config["rag"]
    base_embedder = MultilingualE5Embedder(
        rag["embedding_model"], rag["embedding_revision"],
        passage_prefix=rag["passage_prefix"], query_prefix=rag["query_prefix"]
    )
    offline = offline_benchmark(config, base_embedder, args.sample_size)
    spark = (
        SparkSession.builder.appName("phase-7-rag-efficiency")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    temp_cache = Path("/tmp/phase7-translation-benchmark.sqlite3")
    if temp_cache.exists():
        temp_cache.unlink()
    try:
        curated = "{}/olist".format(config["storage"]["curated_uri"])
        orders = spark.read.parquet("{}/orders_enriched".format(curated)).cache()
        reviews = spark.read.parquet("{}/reviews_enriched".format(curated)).cache()
        review_themes = spark.read.parquet("{}/review_themes".format(curated)).cache()
        orders.count(); reviews.count(); review_themes.count()
        timer = Timer()
        translator = TimedTranslator(
            CachedMarianTranslator(
                rag["translation_model"], rag["translation_revision"], str(temp_cache),
                target_prefix=rag["translation_target_prefix"],
            ),
            timer,
        )
        llm = config["llm"]
        pipeline = ExplanationPipeline(
            TimedAnalyzer(AnalyticsEngine(orders, reviews), timer, "analytics_spark"),
            TimedAnalyzer(ThemeEvidenceEngine(reviews, review_themes), timer, "theme_analytics_spark"),
            ReviewRetriever(
                TimedCollection(get_collection(chroma_client(config), config), timer),
                TimedEmbedder(base_embedder, timer),
                translator,
                reranker=TimedReranker(
                    MultilingualCrossEncoderReranker(
                        rag["reranker_model"],
                        rag["reranker_revision"],
                        batch_size=int(rag["reranker_batch_size"]),
                        max_length=int(rag["reranker_max_length"]),
                    ),
                    timer,
                ),
                candidate_k=int(rag["retrieval_candidate_k"]),
            ),
            TimedGenerator(
                build_generator(llm),
                timer,
            ),
        )
        rows = []
        for repetition in range(1, args.repetitions + 1):
            timer.reset()
            hits_before, misses_before = translator.cache_hits, translator.cache_misses
            end_to_end_started = time.perf_counter()
            started = time.perf_counter()
            question = interpret_question(
                "Perché il rating è diminuito a marzo 2018?",
                start_month="2018-03", end_month="2018-03",
            )
            timer.add("question_interpreter", time.perf_counter() - started)
            pipeline_started = time.perf_counter()
            result = pipeline.run(
                question,
                query_id="phase7-efficiency-{}".format(repetition),
                themes_limit=3,
                evidence_per_theme=1,
                translate=True,
            )
            pipeline_seconds = time.perf_counter() - pipeline_started
            end_to_end = time.perf_counter() - end_to_end_started
            measured = dict(timer.values)
            measured["analytics_spark_total"] = measured.get("analytics_spark", 0.0) + measured.get("theme_analytics_spark", 0.0)
            measured["pipeline_other"] = max(
                0.0,
                pipeline_seconds
                - sum(measured.get(key, 0.0) for key in (
                    "analytics_spark", "theme_analytics_spark", "query_embedding",
                    "chroma_retrieval", "reranking", "translation_top_k", "llm_generation",
                )),
            )
            measured["end_to_end"] = end_to_end
            rows.append(
                {
                    "repetition": repetition,
                    "cache_state": (
                        "sequential_models_translation_cache_cold"
                        if repetition == 1
                        else "sequential_models_reloaded_translation_cache_warm"
                    ),
                    "translation_cache_hits": translator.cache_hits - hits_before,
                    "translation_cache_misses": translator.cache_misses - misses_before,
                    "generation_status": result["generation"]["generation_status"],
                    **measured,
                }
            )
        timing_keys = (
            "question_interpreter", "query_embedding", "chroma_retrieval",
            "reranking", "analytics_spark_total", "translation_top_k", "llm_generation",
            "pipeline_other", "end_to_end",
        )
        result = {
            "schema_version": "1.0",
            "configuration": {
                "hardware": hardware(),
                "embedding_model": rag["embedding_model"],
                "embedding_revision": rag["embedding_revision"],
                "embedding_batch_size": rag["embedding_batch_size"],
                "retrieval_candidate_k": rag["retrieval_candidate_k"],
                "reranker_model": rag["reranker_model"],
                "reranker_revision": rag["reranker_revision"],
                "reranker_batch_size": rag["reranker_batch_size"],
                "llm_model": llm["model"],
                "llm_revision": llm.get("revision"),
                "translation_model": rag["translation_model"],
                "translation_revision": rag["translation_revision"],
                "indexed_documents": get_collection(chroma_client(config), config).count(),
                "themes_limit": 3,
                "evidence_per_theme": 1,
                "effective_max_final_evidence": 3,
                "candidate_k_per_theme": rag["retrieval_candidate_k"],
                "repetitions": args.repetitions,
                "spark_workers": 2,
            },
            "storage": {
                "chroma_index_bytes": args.chroma_bytes,
                "translation_cache_bytes_before_benchmark": args.translation_cache_bytes,
            },
            "offline": offline,
            "online": {
                "repetitions": rows,
                "cold": {key: rows[0][key] for key in timing_keys},
                "warm_summary": summarize(rows[1:], timing_keys),
            },
        }
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "efficiency_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "PHASE7_EFFICIENCY="
            + json.dumps(
                {
                    "full_index_seconds": offline["full_initial_build"]["total_seconds"],
                    "cold_end_to_end_seconds": rows[0]["end_to_end"],
                    "warm_end_to_end_mean_seconds": result["online"]["warm_summary"]["end_to_end"]["mean_seconds"],
                },
                sort_keys=True,
            )
        )
    finally:
        if temp_cache.exists():
            temp_cache.unlink()
        spark.stop()


if __name__ == "__main__":
    main()
