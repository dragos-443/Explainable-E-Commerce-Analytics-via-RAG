"""Collect, blind-annotate and score the 30-query retrieval holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests

from ecommerce_rag.common.config import load_config
from ecommerce_rag.evaluation.embedding_comparison import build_retriever, percentile
from ecommerce_rag.rag.retrieval.service import RetrievalFilters


DATA_ROOT = Path(__file__).with_name("data")
QUERY_PATH = DATA_ROOT / "retrieval_holdout_queries.json"
METHOD = "metadata_plus_semantic_reranked"
ANNOTATION_FILENAME = "annotations_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "annotate", "score", "all"))
    parser.add_argument("--environment", default="local")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--output-root",
        default="/workspace/reports/evaluation/retrieval/holdout-30",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_queries() -> List[dict]:
    return json.loads(QUERY_PATH.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latency_summary(values: Sequence[float]) -> dict:
    return {
        "observations": len(values),
        "mean_seconds": round(statistics.mean(values), 4),
        "median_seconds": round(statistics.median(values), 4),
        "p95_seconds": round(percentile(values, 0.95), 4),
    }


def collect(config: dict, output_root: Path, top_k: int, repetitions: int) -> dict:
    queries = load_queries()
    variant_runs = []
    for variant in ("e5-small", "e5-base"):
        retriever = build_retriever(config, variant)
        first = queries[0]
        # Warm both the embedding model and the shared cross-encoder.
        retriever.retrieve(
            first["question_it"],
            top_k=top_k,
            filters=RetrievalFilters(**first["filters"]),
            use_reranker=True,
        )
        timings = []
        first_results: Dict[str, List[dict]] = {}
        try:
            for repetition in range(repetitions):
                for number, query in enumerate(queries, start=1):
                    started = time.perf_counter()
                    results = retriever.retrieve(
                        query["question_it"],
                        top_k=top_k,
                        filters=RetrievalFilters(**query["filters"]),
                        use_reranker=True,
                    )
                    timings.append(time.perf_counter() - started)
                    if repetition == 0:
                        first_results[query["query_id"]] = [
                            {
                                "review_id": item["review_id"],
                                "document_original": item["document_original"],
                                "review_score": item["metadata"].get("review_score"),
                                "product_category": item["metadata"].get(
                                    "product_category"
                                ),
                                "customer_state": item["metadata"].get(
                                    "customer_state"
                                ),
                                "purchase_month": item["metadata"].get(
                                    "purchase_month"
                                ),
                            }
                            for item in results
                        ]
                    print(
                        "HOLDOUT_PROGRESS={} repetition={}/{} query={}/{}".format(
                            variant,
                            repetition + 1,
                            repetitions,
                            number,
                            len(queries),
                        ),
                        flush=True,
                    )
        finally:
            retriever.release_models()
        variant_runs.append(
            {
                "variant": variant,
                "results": first_results,
                "latency": latency_summary(timings),
            }
        )

    pool = []
    for query in queries:
        query_id = query["query_id"]
        candidates: Dict[str, dict] = {}
        for variant in variant_runs:
            for item in variant["results"][query_id]:
                candidates.setdefault(item["review_id"], item)
        ordered = sorted(
            candidates.values(),
            key=lambda item: hashlib.sha256(
                "{}:{}".format(query_id, item["review_id"]).encode("utf-8")
            ).hexdigest(),
        )
        pool.append(
            {
                "query": query,
                "candidates": ordered,
            }
        )
    payload = {
        "schema_version": "1.0",
        "query_count": len(queries),
        "top_k": top_k,
        "repetitions": repetitions,
        "query_file_sha256": file_sha256(QUERY_PATH),
        "method": METHOD,
        "pooling": "blind union of E5-small and E5-base top-k",
        "variants": variant_runs,
    }
    write_json(output_root / "runs.json", payload)
    write_json(
        output_root / "blind_pool.json",
        {
            "schema_version": "1.0",
            "query_count": len(queries),
            "top_k": top_k,
            "query_file_sha256": file_sha256(QUERY_PATH),
            "pool": pool,
        },
    )
    print(
        "HOLDOUT_COLLECTED="
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


def response_text(payload: dict) -> str:
    texts = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                texts.append(content["text"])
    if not texts:
        raise RuntimeError("OpenAI returned no annotation output")
    return "\n".join(texts)


def annotate_query(config: dict, query: dict, candidates: List[dict]) -> Tuple[List[dict], dict]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for holdout annotation")
    llm = config["llm"]
    ids = [item["review_id"] for item in candidates]
    evidence = [
        {"review_id": item["review_id"], "text_pt": item["document_original"]}
        for item in candidates
    ]
    instructions = (
        "Valuta la rilevanza binaria delle recensioni rispetto alla domanda italiana. "
        "Il testo autorevole è il portoghese. I filtri strutturali di categoria, stato, "
        "periodo e rating sono già soddisfatti e garantiti dai metadati: NON richiedere "
        "che siano ripetuti nel testo e NON rifiutare una recensione perché non nomina "
        "categoria, stato o periodo. Valuta soltanto il problema espresso. Mancata "
        "consegna include un prodotto non ancora ricevuto al momento della recensione; "
        "ritardo include sia una consegna avvenuta tardi sia un prodotto ancora assente "
        "dopo il termine promesso; ordine errato o incompleto include quantità o parti "
        "mancanti; difetto include prodotto rotto, danneggiato o non funzionante; "
        "assistenza o rimborso irrisolto richiede una richiesta/reclamo senza risposta, "
        "soluzione o restituzione del denaro; qualità include fragilità, materiali scarsi "
        "o prodotto non conforme a descrizione e aspettative. relevant=true solo quando "
        "la recensione esprime il problema richiesto; una mera intenzione futura o un "
        "semplice rischio è false. Non inferire il problema se assente. Restituisci "
        "esattamente un giudizio per ciascun review_id. La provenienza dal retriever è nascosta."
    )
    schema = {
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "review_id": {"type": "string", "enum": ids},
                        "relevant": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                    "required": ["review_id", "relevant", "note"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["judgments"],
        "additionalProperties": False,
    }
    request_payload = {
        "model": os.getenv("OPENAI_MODEL", llm["openai_model"]),
        "input": [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question_it": query["question_it"],
                        "expected_aspects": query["expected_aspects"],
                        "reviews": evidence,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "reasoning": {"effort": "low"},
        "max_output_tokens": 1800,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "retrieval_relevance_judgments",
                "strict": True,
                "schema": schema,
            }
        },
    }
    last_error = None
    for _ in range(2):
        try:
            response = requests.post(
                llm["openai_api_url"],
                headers={
                    "Authorization": "Bearer {}".format(api_key),
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=max(120, int(llm["openai_timeout_seconds"])),
            )
            response.raise_for_status()
            payload = response.json()
            judgments = json.loads(response_text(payload))["judgments"]
            returned = [item["review_id"] for item in judgments]
            if len(returned) != len(ids) or set(returned) != set(ids):
                raise ValueError("Annotation response does not cover the blind pool")
            return judgments, payload.get("usage") or {}
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            last_error = exc
    raise RuntimeError("Holdout annotation failed after two attempts") from last_error


def annotate(config: dict, output_root: Path) -> dict:
    blind_pool = json.loads(
        (output_root / "blind_pool.json").read_text(encoding="utf-8")
    )
    output_path = output_root / ANNOTATION_FILENAME
    if output_path.exists():
        result = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        result = {
            "schema_version": "1.0",
            "annotation_method": (
                "blind binary relevance with a multilingual LLM; protocol v2 treats "
                "category, state, period and rating as guaranteed metadata"
            ),
            "model": os.getenv("OPENAI_MODEL", config["llm"]["openai_model"]),
            "query_file_sha256": blind_pool["query_file_sha256"],
            "queries": {},
            "usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
            },
        }
    for number, item in enumerate(blind_pool["pool"], start=1):
        query_id = item["query"]["query_id"]
        expected_ids = {candidate["review_id"] for candidate in item["candidates"]}
        existing = result["queries"].get(query_id)
        if existing and {row["review_id"] for row in existing} == expected_ids:
            print("HOLDOUT_ANNOTATION_SKIPPED={}".format(query_id), flush=True)
            continue
        judgments, usage = annotate_query(config, item["query"], item["candidates"])
        result["queries"][query_id] = judgments
        details = usage.get("input_tokens_details") or {}
        result["usage"]["input_tokens"] += int(usage.get("input_tokens") or 0)
        result["usage"]["cached_input_tokens"] += int(details.get("cached_tokens") or 0)
        result["usage"]["output_tokens"] += int(usage.get("output_tokens") or 0)
        write_json(output_path, result)
        print(
            "HOLDOUT_ANNOTATED={}/{} {}".format(
                number, len(blind_pool["pool"]), query_id
            ),
            flush=True,
        )
    return result


def reciprocal_rank(ranked: Sequence[str], relevant: set) -> float:
    for rank, review_id in enumerate(ranked, start=1):
        if review_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
    gains = [1.0 / math.log2(rank + 1) for rank, item in enumerate(ranked[:k], 1) if item in relevant]
    ideal_count = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return sum(gains) / ideal if ideal else 0.0


def confidence_interval(values: Sequence[float], seed: int = 42) -> List[float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(10000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(statistics.mean(sample))
    samples.sort()
    return [round(samples[249], 4), round(samples[9749], 4)]


def summarize_rows(rows: List[dict], variant: str) -> dict:
    selected = [row for row in rows if row["variant"] == variant]
    result = {}
    for metric in (
        "precision_at_k",
        "recall_in_pool_at_k",
        "hit_rate_at_k",
        "mrr_at_k",
        "ndcg_at_k",
    ):
        values = [row[metric] for row in selected if row[metric] is not None]
        result[metric] = round(statistics.mean(values), 4) if values else None
    result["queries_with_relevant_in_pool"] = sum(
        row["relevant_in_pool"] > 0 for row in selected
    )
    result["queries_without_relevant_in_pool"] = sum(
        row["relevant_in_pool"] == 0 for row in selected
    )
    result["precision_at_k_95pct_bootstrap_ci"] = confidence_interval(
        [row["precision_at_k"] for row in selected]
    )
    return result


def combine_with_development(output_root: Path, holdout_rows: List[dict]) -> dict | None:
    development_path = (
        output_root.parent / "e5-base-comparison" / "adjudicated_metrics.json"
    )
    if not development_path.exists():
        return None
    development = json.loads(development_path.read_text(encoding="utf-8"))
    rows = []
    for row in development["per_query"]:
        rows.append(
            {
                "query_id": "development/{}".format(row["query_id"]),
                "variant": row["variant"],
                "precision_at_k": row["precision_at_k"],
                "recall_in_pool_at_k": row["recall_in_two_model_pool_at_k"],
                "hit_rate_at_k": row["hit_rate_at_k"],
            }
        )
    for row in holdout_rows:
        rows.append(
            {
                "query_id": "holdout/{}".format(row["query_id"]),
                "variant": row["variant"],
                "precision_at_k": row["precision_at_k"],
                "recall_in_pool_at_k": row["recall_in_pool_at_k"],
                "hit_rate_at_k": row["hit_rate_at_k"],
            }
        )
    summary = {}
    for variant in ("e5-small", "e5-base"):
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {}
        for metric in (
            "precision_at_k",
            "recall_in_pool_at_k",
            "hit_rate_at_k",
        ):
            values = [row[metric] for row in selected if row[metric] is not None]
            summary[variant][metric] = round(statistics.mean(values), 4)
        summary[variant]["precision_at_k_95pct_bootstrap_ci"] = confidence_interval(
            [row["precision_at_k"] for row in selected]
        )
    by_query: Dict[str, Dict[str, float]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], {})[row["variant"]] = row[
            "precision_at_k"
        ]
    differences = [
        values["e5-base"] - values["e5-small"] for values in by_query.values()
    ]
    result = {
        "schema_version": "1.0",
        "query_count": len(by_query),
        "development_queries": development["query_count"],
        "holdout_queries": len({row["query_id"] for row in holdout_rows}),
        "method": METHOD,
        "warning": (
            "The 20 development queries influenced earlier design choices. The "
            "30-query holdout remains the primary generalization estimate."
        ),
        "summary": summary,
        "paired_precision_difference_base_minus_small": round(
            statistics.mean(differences), 4
        ),
        "paired_difference_95pct_bootstrap_ci": confidence_interval(differences),
    }
    write_json(output_root / "combined_50_metrics.json", result)
    return result


def score(output_root: Path) -> dict:
    runs = json.loads((output_root / "runs.json").read_text(encoding="utf-8"))
    annotations = json.loads(
        (output_root / ANNOTATION_FILENAME).read_text(encoding="utf-8")
    )
    if annotations["query_file_sha256"] != runs["query_file_sha256"]:
        raise ValueError("Annotations and retrieval runs use different query files")
    variants = {item["variant"]: item for item in runs["variants"]}
    aspects = {
        query["query_id"]: query["expected_aspects"][0] for query in load_queries()
    }
    query_ids = set(variants["e5-small"]["results"])
    if set(annotations["queries"]) != query_ids:
        raise ValueError("Annotations do not cover all holdout queries")
    rows = []
    outcomes = {"e5-base_wins": 0, "ties": 0, "e5-small_wins": 0}
    for query_id in sorted(query_ids):
        relevant = {
            item["review_id"]
            for item in annotations["queries"][query_id]
            if item["relevant"]
        }
        pool_ids = set()
        ranked_by_variant = {}
        for variant in ("e5-small", "e5-base"):
            ranked = [
                item["review_id"] for item in variants[variant]["results"][query_id]
            ]
            ranked_by_variant[variant] = ranked
            pool_ids.update(ranked)
        annotation_ids = [
            item["review_id"] for item in annotations["queries"][query_id]
        ]
        if (
            len(annotation_ids) != len(set(annotation_ids))
            or set(annotation_ids) != pool_ids
        ):
            raise ValueError(
                "Annotations do not exactly cover the pooled ids for {}".format(
                    query_id
                )
            )
        if not relevant.issubset(pool_ids):
            raise ValueError("Unknown relevant id for {}".format(query_id))
        precision_by_variant = {}
        for variant, ranked in ranked_by_variant.items():
            hits = len(set(ranked) & relevant)
            precision = hits / runs["top_k"]
            precision_by_variant[variant] = precision
            rows.append(
                {
                    "query_id": query_id,
                    "aspect": aspects[query_id],
                    "variant": variant,
                    "relevant_at_k": hits,
                    "relevant_in_pool": len(relevant),
                    "precision_at_k": precision,
                    "recall_in_pool_at_k": hits / len(relevant) if relevant else None,
                    "hit_rate_at_k": float(hits > 0),
                    "mrr_at_k": reciprocal_rank(ranked, relevant),
                    "ndcg_at_k": ndcg_at_k(ranked, relevant, runs["top_k"]),
                }
            )
        difference = precision_by_variant["e5-base"] - precision_by_variant["e5-small"]
        if difference > 0:
            outcomes["e5-base_wins"] += 1
        elif difference < 0:
            outcomes["e5-small_wins"] += 1
        else:
            outcomes["ties"] += 1
    summary = {
        variant: summarize_rows(rows, variant) for variant in ("e5-small", "e5-base")
    }
    by_aspect = {}
    for aspect in sorted(set(aspects.values())):
        aspect_rows = [row for row in rows if row["aspect"] == aspect]
        by_aspect[aspect] = {
            variant: summarize_rows(aspect_rows, variant)
            for variant in ("e5-small", "e5-base")
        }
    differences = []
    for query_id in sorted(query_ids):
        base = next(
            row for row in rows if row["query_id"] == query_id and row["variant"] == "e5-base"
        )
        small = next(
            row for row in rows if row["query_id"] == query_id and row["variant"] == "e5-small"
        )
        differences.append(base["precision_at_k"] - small["precision_at_k"])
    result = {
        "schema_version": "1.0",
        "query_count": len(query_ids),
        "top_k": runs["top_k"],
        "method": runs["method"],
        "annotation_method": annotations["annotation_method"],
        "annotation_model": annotations["model"],
        "judged_pairs": sum(len(items) for items in annotations["queries"].values()),
        "summary": summary,
        "by_aspect": by_aspect,
        "paired_precision_difference_base_minus_small": round(
            statistics.mean(differences), 4
        ),
        "paired_difference_95pct_bootstrap_ci": confidence_interval(differences),
        "paired_query_outcomes": outcomes,
        "latency": {
            variant: variants[variant]["latency"]
            for variant in ("e5-small", "e5-base")
        },
        "usage": annotations["usage"],
        "per_query": rows,
    }
    combined = combine_with_development(output_root, rows)
    if combined is not None:
        result["combined_50_summary"] = combined["summary"]
    write_json(output_root / "metrics.json", result)
    print("HOLDOUT_METRICS=" + json.dumps(result["summary"], sort_keys=True), flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.repetitions < 1:
        raise ValueError("top-k and repetitions must be positive")
    config = load_config(args.environment)
    output_root = Path(args.output_root)
    if args.mode in ("collect", "all"):
        collect(config, output_root, args.top_k, args.repetitions)
    if args.mode in ("annotate", "all"):
        annotate(config, output_root)
    if args.mode in ("score", "all"):
        score(output_root)


if __name__ == "__main__":
    main()
