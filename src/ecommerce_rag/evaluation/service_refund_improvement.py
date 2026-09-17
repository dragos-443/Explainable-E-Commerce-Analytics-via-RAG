"""Exploratory evaluation of stricter retrieval for unresolved service/refunds."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

from ecommerce_rag.common.config import load_config
from ecommerce_rag.evaluation.embedding_comparison import build_retriever
from ecommerce_rag.evaluation.retrieval_holdout import (
    annotate_query,
    confidence_interval,
    load_queries,
    ndcg_at_k,
    reciprocal_rank,
    write_json,
)
from ecommerce_rag.rag.explanation_pipeline import theme_retrieval_query
from ecommerce_rag.rag.retrieval.service import RetrievalFilters


THEME = "service_or_refund"
LEGACY_DESCRIPTION = "assistenza senza risposta, problema non risolto, reso o rimborso"
DEFAULT_BASELINE = Path(
    "/workspace/reports/evaluation/retrieval/holdout-30/runs.json"
)
DEFAULT_OUTPUT = Path(
    "/workspace/reports/evaluation/diagnostics/service-refund-query"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "annotate", "score", "all"))
    parser.add_argument("--environment", default="local")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def service_queries() -> List[dict]:
    return [
        query
        for query in load_queries()
        if THEME in query["expected_aspects"]
    ]


def compact_result(item: dict) -> dict:
    return {
        "review_id": item["review_id"],
        "document_original": item["document_original"],
        "review_score": item["metadata"].get("review_score"),
        "product_category": item["metadata"].get("product_category"),
        "customer_state": item["metadata"].get("customer_state"),
        "purchase_month": item["metadata"].get("purchase_month"),
    }


def legacy_retrieval_query(question: str) -> str:
    return "{} Cerca recensioni che descrivono: {}.".format(
        question, LEGACY_DESCRIPTION
    )


def collect(
    config: dict,
    baseline_path: Path,
    output_root: Path,
    top_k: int,
    repetitions: int,
) -> dict:
    queries = service_queries()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_variants = {item["variant"]: item for item in baseline["variants"]}
    variants = []
    for encoder in ("e5-small", "e5-base"):
        raw = {
            query["query_id"]: baseline_variants[encoder]["results"][query["query_id"]]
            for query in queries
        }
        variants.append({"variant": encoder + "-raw", "results": raw})

        retriever = build_retriever(config, encoder)
        timings = {"focused": [], "legacy-production": [], "production": []}
        results: Dict[str, Dict[str, List[dict]]] = {
            "focused": {},
            "legacy-production": {},
            "production": {},
        }
        first = queries[0]
        retriever.retrieve(
            theme_retrieval_query(first["question_it"], THEME),
            top_k=top_k,
            filters=RetrievalFilters(**first["filters"]),
            use_reranker=True,
        )
        try:
            for repetition in range(repetitions):
                for query in queries:
                    focused = theme_retrieval_query(query["question_it"], THEME)
                    for strategy, retrieval_query, use_theme_filter in (
                        ("focused", focused, False),
                        (
                            "legacy-production",
                            legacy_retrieval_query(query["question_it"]),
                            True,
                        ),
                        ("production", focused, True),
                    ):
                        filters = dict(query["filters"])
                        if use_theme_filter:
                            filters["theme"] = THEME
                        started = time.perf_counter()
                        selected = retriever.retrieve(
                            retrieval_query,
                            top_k=top_k,
                            filters=RetrievalFilters(**filters),
                            use_reranker=True,
                        )
                        timings[strategy].append(time.perf_counter() - started)
                        if repetition == 0:
                            results[strategy][query["query_id"]] = [
                                compact_result(item) for item in selected
                            ]
        finally:
            retriever.release_models()
        for strategy in ("focused", "legacy-production", "production"):
            values = timings[strategy]
            variants.append(
                {
                    "variant": encoder + "-" + strategy,
                    "results": results[strategy],
                    "latency_seconds": {
                        "observations": len(values),
                        "mean": round(statistics.mean(values), 4),
                        "median": round(statistics.median(values), 4),
                    },
                }
            )

    pool = []
    for query in queries:
        candidates = {}
        for variant in variants:
            for item in variant["results"][query["query_id"]]:
                candidates.setdefault(item["review_id"], item)
        pool.append({"query": query, "candidates": list(candidates.values())})
    payload = {
        "schema_version": "1.0",
        "status": "exploratory_after_holdout_inspection",
        "theme": THEME,
        "query_count": len(queries),
        "top_k": top_k,
        "repetitions": repetitions,
        "query_focus": theme_retrieval_query("<domanda>", THEME),
        "variants": variants,
        "pool": pool,
    }
    write_json(output_root / "runs.json", payload)
    print(
        "SERVICE_REFUND_COLLECTED="
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


def annotate(config: dict, output_root: Path) -> dict:
    runs = json.loads((output_root / "runs.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "1.0",
        "status": runs["status"],
        "model": config["llm"]["openai_model"],
        "queries": {},
        "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
    }
    for item in runs["pool"]:
        query_id = item["query"]["query_id"]
        judgments, usage = annotate_query(
            config, item["query"], item["candidates"]
        )
        result["queries"][query_id] = judgments
        details = usage.get("input_tokens_details") or {}
        result["usage"]["input_tokens"] += int(usage.get("input_tokens") or 0)
        result["usage"]["cached_input_tokens"] += int(
            details.get("cached_tokens") or 0
        )
        result["usage"]["output_tokens"] += int(usage.get("output_tokens") or 0)
        write_json(output_root / "annotations.json", result)
        print("SERVICE_REFUND_ANNOTATED=" + query_id, flush=True)
    return result


def summarize(rows: List[dict], variant: str) -> dict:
    selected = [row for row in rows if row["variant"] == variant]
    result = {}
    for metric in ("precision_at_5", "recall_in_pool_at_5", "hit_rate_at_5", "mrr_at_5", "ndcg_at_5"):
        values = [row[metric] for row in selected if row[metric] is not None]
        result[metric] = round(statistics.mean(values), 4) if values else None
    result["precision_at_5_95pct_bootstrap_ci"] = confidence_interval(
        [row["precision_at_5"] for row in selected]
    )
    return result


def score(output_root: Path) -> dict:
    runs = json.loads((output_root / "runs.json").read_text(encoding="utf-8"))
    annotation_path = output_root / "annotations.json"
    if annotation_path.exists():
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        relevant_by_query = {
            query_id: {
                item["review_id"] for item in judgments if item["relevant"]
            }
            for query_id, judgments in annotations["queries"].items()
        }
        annotation_source = "blind_llm"
        usage = annotations["usage"]
    else:
        adjudication_path = output_root / "assistant_adjudication.json"
        annotations = json.loads(adjudication_path.read_text(encoding="utf-8"))
        relevant_by_query = {
            query_id: set(review_ids)
            for query_id, review_ids in annotations["relevant_ids"].items()
        }
        annotation_source = annotations["annotation_source"]
        usage = None
    expected_queries = {item["query"]["query_id"] for item in runs["pool"]}
    if set(relevant_by_query) != expected_queries:
        raise ValueError("Adjudication does not cover every experiment query")
    pool_ids = {
        item["query"]["query_id"]: {
            candidate["review_id"] for candidate in item["candidates"]
        }
        for item in runs["pool"]
    }
    for query_id, relevant in relevant_by_query.items():
        if not relevant.issubset(pool_ids[query_id]):
            raise ValueError("Adjudication contains an unknown review id")
    rows = []
    for variant in runs["variants"]:
        for query_id, results in variant["results"].items():
            relevant = relevant_by_query[query_id]
            ranked = [item["review_id"] for item in results]
            found = len(set(ranked) & relevant)
            rows.append(
                {
                    "query_id": query_id,
                    "variant": variant["variant"],
                    "relevant_at_5": found,
                    "relevant_in_pool": len(relevant),
                    "precision_at_5": found / runs["top_k"],
                    "recall_in_pool_at_5": found / len(relevant) if relevant else None,
                    "hit_rate_at_5": float(found > 0),
                    "mrr_at_5": reciprocal_rank(ranked, relevant),
                    "ndcg_at_5": ndcg_at_k(ranked, relevant, runs["top_k"]),
                }
            )
    names = [item["variant"] for item in runs["variants"]]
    summary = {name: summarize(rows, name) for name in names}
    improvements = {}
    for encoder in ("e5-small", "e5-base"):
        raw = summary[encoder + "-raw"]["precision_at_5"]
        focused = summary[encoder + "-focused"]["precision_at_5"]
        legacy = summary[encoder + "-legacy-production"]["precision_at_5"]
        production = summary[encoder + "-production"]["precision_at_5"]
        improvements[encoder] = {
            "focused_minus_raw": round(focused - raw, 4),
            "production_minus_raw": round(production - raw, 4),
            "production_minus_legacy_production": round(production - legacy, 4),
        }
    payload = {
        "schema_version": "1.0",
        "status": runs["status"],
        "theme": THEME,
        "query_count": runs["query_count"],
        "judged_pairs": sum(len(item["candidates"]) for item in runs["pool"]),
        "annotation_source": annotation_source,
        "summary": summary,
        "precision_at_5_differences": improvements,
        "per_query": rows,
        "usage": usage,
    }
    write_json(output_root / "metrics.json", payload)
    print("SERVICE_REFUND_METRICS=" + json.dumps(payload["summary"], sort_keys=True))
    return payload


def main() -> None:
    args = parse_args()
    if args.top_k != 5:
        raise ValueError("This experiment currently reports metrics at k=5")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    config = load_config(args.environment)
    if args.mode in ("collect", "all"):
        collect(config, args.baseline, args.output_root, args.top_k, args.repetitions)
    if args.mode in ("annotate", "all"):
        annotate(config, args.output_root)
    if args.mode in ("score", "all"):
        score(args.output_root)


if __name__ == "__main__":
    main()
