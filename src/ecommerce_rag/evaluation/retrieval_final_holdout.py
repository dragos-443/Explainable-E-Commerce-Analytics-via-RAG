"""Evaluate E5-small on 50 frozen queries and summarize all 100 queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence

from ecommerce_rag.common.config import load_config
from ecommerce_rag.evaluation.embedding_comparison import build_retriever
from ecommerce_rag.evaluation.retrieval_holdout import (
    annotate_query,
    confidence_interval,
    file_sha256,
    latency_summary,
    ndcg_at_k,
    reciprocal_rank,
    write_json,
)
from ecommerce_rag.rag.retrieval.service import RetrievalFilters


DATA_ROOT = Path(__file__).with_name("data")
QUERY_PATH = DATA_ROOT / "retrieval_final_holdout_queries.json"
DEFAULT_OUTPUT = Path("/workspace/reports/evaluation/diagnostics/stress-test-50")
DEVELOPMENT_METRICS = Path(
    "/workspace/reports/evaluation/comparisons/e5-base/adjudicated_metrics.json"
)
FIRST_HOLDOUT_METRICS = Path(
    "/workspace/reports/evaluation/retrieval/holdout-30/metrics.json"
)
METHOD = "e5-small metadata filters plus semantic retrieval and reranker"
BENCHMARK_STATUS = "frozen_final_holdout"
COMBINED_NEW_LABEL = "final_holdout"
COMBINED_WARNING = (
    "The 20 development queries influenced earlier design choices. "
    "Use the frozen final 50-query holdout as the primary new estimate."
)
ALLOW_ANNOTATION_REPLACEMENT = False
CODEX_ADJUDICATION_FILENAME = "codex_adjudication.json"
ANNOTATION_FILENAMES = {
    "codex": "annotations_codex.json",
    "openai": "annotations_openai.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "annotate", "score", "all"))
    parser.add_argument("--environment", default="local")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--annotation-source",
        choices=("auto", "openai", "codex"),
        default="auto",
    )
    return parser.parse_args()


def load_queries() -> List[dict]:
    return json.loads(QUERY_PATH.read_text(encoding="utf-8"))


def retrieval_query(query: dict) -> str:
    """Return the text sent to dense retrieval and reranking."""
    return query["question_it"]


def compact_result(item: dict) -> dict:
    return {
        "review_id": item["review_id"],
        "document_original": item["document_original"],
        "review_score": item["metadata"].get("review_score"),
        "product_category": item["metadata"].get("product_category"),
        "customer_state": item["metadata"].get("customer_state"),
        "purchase_month": item["metadata"].get("purchase_month"),
    }


def collect(config: dict, output_root: Path, top_k: int, repetitions: int) -> dict:
    queries = load_queries()
    candidate_k = int(config["rag"]["retrieval_candidate_k"])
    if candidate_k != 50:
        raise ValueError("The final benchmark requires retrieval_candidate_k=50")
    retriever = build_retriever(config, "e5-small")
    first = queries[0]
    retriever.retrieve(
        retrieval_query(first),
        top_k=top_k,
        filters=RetrievalFilters(**first["filters"]),
        use_reranker=True,
    )
    timings: List[float] = []
    results: Dict[str, List[dict]] = {}
    try:
        for repetition in range(repetitions):
            for number, query in enumerate(queries, start=1):
                started = time.perf_counter()
                selected = retriever.retrieve(
                    retrieval_query(query),
                    top_k=top_k,
                    filters=RetrievalFilters(**query["filters"]),
                    use_reranker=True,
                )
                timings.append(time.perf_counter() - started)
                if len(selected) != top_k:
                    raise ValueError(
                        "Expected {} final results for {}, found {}".format(
                            top_k, query["query_id"], len(selected)
                        )
                    )
                if repetition == 0:
                    results[query["query_id"]] = [
                        compact_result(item) for item in selected
                    ]
                print(
                    "FINAL_HOLDOUT_PROGRESS repetition={}/{} query={}/{}".format(
                        repetition + 1, repetitions, number, len(queries)
                    ),
                    flush=True,
                )
    finally:
        retriever.release_models()

    pool = []
    for query in queries:
        query_id = query["query_id"]
        ordered = sorted(
            results[query_id],
            key=lambda item: hashlib.sha256(
                "{}:{}".format(query_id, item["review_id"]).encode("utf-8")
            ).hexdigest(),
        )
        pool.append({"query": query, "candidates": ordered})
    payload = {
        "schema_version": "1.0",
        "status": BENCHMARK_STATUS,
        "query_count": len(queries),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "repetitions": repetitions,
        "query_file_sha256": file_sha256(QUERY_PATH),
        "model": "intfloat/multilingual-e5-small",
        "method": METHOD,
        "results": results,
        "latency": latency_summary(timings),
    }
    write_json(output_root / "runs.json", payload)
    write_json(
        output_root / "blind_pool.json",
        {
            "schema_version": "1.0",
            "status": payload["status"],
            "query_count": len(queries),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "query_file_sha256": payload["query_file_sha256"],
            "pool": pool,
        },
    )
    print(
        "FINAL_HOLDOUT_COLLECTED="
        + json.dumps(
            {
                "queries": len(queries),
                "judgments_required": sum(len(item["candidates"]) for item in pool),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def annotation_source(output_root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if (output_root / ANNOTATION_FILENAMES["openai"]).exists():
        return "openai"
    return "codex"


def annotate(config: dict, output_root: Path, requested_source: str) -> dict:
    pool = json.loads((output_root / "blind_pool.json").read_text(encoding="utf-8"))
    source = annotation_source(output_root, requested_source)
    output_path = output_root / ANNOTATION_FILENAMES[source]
    codex_path = output_root / CODEX_ADJUDICATION_FILENAME
    if source == "codex":
        if not codex_path.exists():
            raise ValueError("Codex adjudication file is not available")
        adjudication = json.loads(codex_path.read_text(encoding="utf-8"))
        if adjudication["query_file_sha256"] != pool["query_file_sha256"]:
            raise ValueError("Codex adjudication refers to a different query file")
        expected_queries = {
            entry["query"]["query_id"] for entry in pool["pool"]
        }
        if set(adjudication["relevant_ids"]) != expected_queries:
            raise ValueError("Codex adjudication does not cover exactly 50 queries")
        result = {
            "schema_version": "1.0",
            "status": pool["status"],
            "annotation_method": adjudication["annotation_method"],
            "model": "Codex interactive adjudication",
            "query_file_sha256": pool["query_file_sha256"],
            "queries": {},
            "usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
            },
        }
        for entry in pool["pool"]:
            query_id = entry["query"]["query_id"]
            expected_ids = {item["review_id"] for item in entry["candidates"]}
            relevant_ids = set(adjudication["relevant_ids"][query_id])
            if not relevant_ids.issubset(expected_ids):
                raise ValueError("Unknown relevant review for " + query_id)
            result["queries"][query_id] = [
                {
                    "review_id": item["review_id"],
                    "relevant": item["review_id"] in relevant_ids,
                    "note": "Blind Codex relevance judgment after API HTTP 429.",
                }
                for item in entry["candidates"]
            ]
        write_json(output_path, result)
        print(
            "FINAL_HOLDOUT_CODEX_ADJUDICATED="
            + json.dumps(
                {
                    "queries": len(result["queries"]),
                    "judgments": sum(len(items) for items in result["queries"].values()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return result
    create_annotations = not output_path.exists()
    if output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if result["query_file_sha256"] != pool["query_file_sha256"]:
            if not ALLOW_ANNOTATION_REPLACEMENT:
                raise ValueError("Existing annotations refer to a different query file")
            create_annotations = True
    if create_annotations:
        result = {
            "schema_version": "1.0",
            "status": pool["status"],
            "annotation_method": (
                "blind binary relevance over E5-small top-5; structured metadata "
                "constraints are treated as guaranteed"
            ),
            "model": os.getenv("OPENAI_MODEL", config["llm"]["openai_model"]),
            "query_file_sha256": pool["query_file_sha256"],
            "queries": {},
            "usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
            },
        }
    for number, entry in enumerate(pool["pool"], start=1):
        query_id = entry["query"]["query_id"]
        expected_ids = {item["review_id"] for item in entry["candidates"]}
        existing = result["queries"].get(query_id)
        if existing and {item["review_id"] for item in existing} == expected_ids:
            print("FINAL_HOLDOUT_ANNOTATION_SKIPPED=" + query_id, flush=True)
            continue
        judgments, usage = annotate_query(config, entry["query"], entry["candidates"])
        result["queries"][query_id] = judgments
        details = usage.get("input_tokens_details") or {}
        result["usage"]["input_tokens"] += int(usage.get("input_tokens") or 0)
        result["usage"]["cached_input_tokens"] += int(
            details.get("cached_tokens") or 0
        )
        result["usage"]["output_tokens"] += int(usage.get("output_tokens") or 0)
        write_json(output_path, result)
        print(
            "FINAL_HOLDOUT_ANNOTATED={}/{} {}".format(
                number, len(pool["pool"]), query_id
            ),
            flush=True,
        )
    return result


def summarize(rows: Sequence[dict]) -> dict:
    return {
        "query_count": len(rows),
        "precision_at_5": round(
            statistics.mean(item["precision_at_5"] for item in rows), 4
        ),
        "precision_at_5_95pct_bootstrap_ci": confidence_interval(
            [item["precision_at_5"] for item in rows]
        ),
        "hit_rate_at_5": round(
            statistics.mean(item["hit_rate_at_5"] for item in rows), 4
        ),
        "mrr_at_5": round(statistics.mean(item["mrr_at_5"] for item in rows), 4),
        "ndcg_at_5": round(statistics.mean(item["ndcg_at_5"] for item in rows), 4),
        "queries_without_relevant_top_5": sum(
            item["relevant_at_5"] == 0 for item in rows
        ),
    }


def prior_rows() -> tuple[List[dict], List[dict]]:
    development = json.loads(DEVELOPMENT_METRICS.read_text(encoding="utf-8"))
    first_holdout = json.loads(FIRST_HOLDOUT_METRICS.read_text(encoding="utf-8"))
    development_rows = [
        {
            "query_id": item["query_id"],
            "precision_at_5": item["precision_at_k"],
            "hit_rate_at_5": item["hit_rate_at_k"],
        }
        for item in development["per_query"]
        if item["variant"] == "e5-small"
    ]
    holdout_rows = [
        {
            "query_id": item["query_id"],
            "precision_at_5": item["precision_at_k"],
            "hit_rate_at_5": item["hit_rate_at_k"],
        }
        for item in first_holdout["per_query"]
        if item["variant"] == "e5-small"
    ]
    return development_rows, holdout_rows


def combined_summary(new_rows: List[dict]) -> dict:
    development, first_holdout = prior_rows()
    compact_new = [
        {
            "query_id": item["query_id"],
            "precision_at_5": item["precision_at_5"],
            "hit_rate_at_5": item["hit_rate_at_5"],
        }
        for item in new_rows
    ]
    query_ids = [
        item["query_id"] for item in development + first_holdout + compact_new
    ]
    if len(query_ids) != 100 or len(set(query_ids)) != 100:
        raise ValueError("The combined benchmark must contain 100 unique queries")

    def compact_metrics(rows: Sequence[dict]) -> dict:
        return {
            "query_count": len(rows),
            "precision_at_5": round(
                statistics.mean(item["precision_at_5"] for item in rows), 4
            ),
            "precision_at_5_95pct_bootstrap_ci": confidence_interval(
                [item["precision_at_5"] for item in rows]
            ),
            "hit_rate_at_5": round(
                statistics.mean(item["hit_rate_at_5"] for item in rows), 4
            ),
        }

    all_rows = development + first_holdout + compact_new
    all_holdout = first_holdout + compact_new
    return {
        "schema_version": "1.0",
        "status": "combined_descriptive_summary",
        "candidate_k": 50,
        "top_k": 5,
        "model": "intfloat/multilingual-e5-small",
        "composition": {
            "development": len(development),
            "first_holdout": len(first_holdout),
            COMBINED_NEW_LABEL: len(compact_new),
        },
        "all_100": compact_metrics(all_rows),
        "holdout_80": compact_metrics(all_holdout),
        "warning": COMBINED_WARNING,
    }


def annotation_agreement(primary: dict, secondary: dict) -> dict:
    if primary["query_file_sha256"] != secondary["query_file_sha256"]:
        raise ValueError("Annotation sets refer to different query files")
    primary_pairs = {
        (query_id, item["review_id"]): bool(item["relevant"])
        for query_id, items in primary["queries"].items()
        for item in items
    }
    secondary_pairs = {
        (query_id, item["review_id"]): bool(item["relevant"])
        for query_id, items in secondary["queries"].items()
        for item in items
    }
    if set(primary_pairs) != set(secondary_pairs):
        raise ValueError("Annotation sets do not cover the same query-review pairs")
    both_relevant = sum(
        primary_pairs[key] and secondary_pairs[key] for key in primary_pairs
    )
    both_not_relevant = sum(
        not primary_pairs[key] and not secondary_pairs[key] for key in primary_pairs
    )
    primary_only = sum(
        primary_pairs[key] and not secondary_pairs[key] for key in primary_pairs
    )
    secondary_only = sum(
        not primary_pairs[key] and secondary_pairs[key] for key in primary_pairs
    )
    count = len(primary_pairs)
    observed = (both_relevant + both_not_relevant) / count
    primary_positive = both_relevant + primary_only
    secondary_positive = both_relevant + secondary_only
    expected = (
        primary_positive * secondary_positive
        + (count - primary_positive) * (count - secondary_positive)
    ) / (count * count)
    kappa = (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return {
        "judged_pairs": count,
        "agreement_rate": round(observed, 4),
        "cohens_kappa": round(kappa, 4),
        "both_relevant": both_relevant,
        "both_not_relevant": both_not_relevant,
        "primary_only_relevant": primary_only,
        "secondary_only_relevant": secondary_only,
    }


def score(output_root: Path, requested_source: str) -> dict:
    runs = json.loads((output_root / "runs.json").read_text(encoding="utf-8"))
    source = annotation_source(output_root, requested_source)
    annotations = json.loads(
        (output_root / ANNOTATION_FILENAMES[source]).read_text(encoding="utf-8")
    )
    if annotations["query_file_sha256"] != runs["query_file_sha256"]:
        raise ValueError("Annotations and runs use different frozen query files")
    queries = {item["query_id"]: item for item in load_queries()}
    if set(annotations["queries"]) != set(queries):
        raise ValueError("Annotations do not cover all final holdout queries")
    rows = []
    for query_id in sorted(queries):
        ranked = [item["review_id"] for item in runs["results"][query_id]]
        judgments = annotations["queries"][query_id]
        annotation_ids = [item["review_id"] for item in judgments]
        if len(annotation_ids) != len(set(annotation_ids)) or set(annotation_ids) != set(
            ranked
        ):
            raise ValueError("Annotations do not exactly cover " + query_id)
        relevant = {
            item["review_id"] for item in judgments if bool(item["relevant"])
        }
        hits = len(relevant)
        rows.append(
            {
                "query_id": query_id,
                "aspect": queries[query_id]["expected_aspects"][0],
                "relevant_at_5": hits,
                "precision_at_5": hits / runs["top_k"],
                "hit_rate_at_5": float(hits > 0),
                "mrr_at_5": reciprocal_rank(ranked, relevant),
                "ndcg_at_5": ndcg_at_k(ranked, relevant, runs["top_k"]),
            }
        )
    by_aspect = {
        aspect: summarize([item for item in rows if item["aspect"] == aspect])
        for aspect in sorted({item["aspect"] for item in rows})
    }
    combined = combined_summary(rows)
    payload = {
        "schema_version": "1.0",
        "status": runs["status"],
        "query_count": runs["query_count"],
        "candidate_k": runs["candidate_k"],
        "top_k": runs["top_k"],
        "model": runs["model"],
        "method": runs["method"],
        "annotation_method": annotations["annotation_method"],
        "annotation_model": annotations["model"],
        "judged_pairs": sum(len(items) for items in annotations["queries"].values()),
        "summary": summarize(rows),
        "by_aspect": by_aspect,
        "latency": runs["latency"],
        "usage": annotations["usage"],
        "per_query": rows,
        "combined_100": combined,
    }
    codex_annotations_path = output_root / ANNOTATION_FILENAMES["codex"]
    if source == "openai" and codex_annotations_path.exists():
        codex_annotations = json.loads(
            codex_annotations_path.read_text(encoding="utf-8")
        )
        payload["agreement_with_codex_adjudication"] = annotation_agreement(
            annotations, codex_annotations
        )
    write_json(output_root / "metrics.json", payload)
    write_json(output_root / "combined_100_metrics.json", combined)
    print("FINAL_HOLDOUT_METRICS=" + json.dumps(payload["summary"]), flush=True)
    print("COMBINED_100_METRICS=" + json.dumps(combined["all_100"]), flush=True)
    return payload


def main() -> None:
    args = parse_args()
    if args.top_k != 5:
        raise ValueError("The final benchmark is fixed at top_k=5")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    config = load_config(args.environment)
    if args.mode in ("collect", "all"):
        collect(config, args.output_root, args.top_k, args.repetitions)
    if args.mode in ("annotate", "all"):
        annotate(config, args.output_root, args.annotation_source)
    if args.mode in ("score", "all"):
        score(args.output_root, args.annotation_source)


if __name__ == "__main__":
    main()
