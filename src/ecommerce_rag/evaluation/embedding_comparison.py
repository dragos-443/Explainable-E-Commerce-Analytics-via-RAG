"""Reproducible, isolated comparison of multilingual E5 embedding models."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ecommerce_rag.common.config import load_config
from ecommerce_rag.rag.embeddings.model import MultilingualE5Embedder
from ecommerce_rag.rag.index import chroma_client, get_collection
from ecommerce_rag.rag.retrieval.reranking import MultilingualCrossEncoderReranker
from ecommerce_rag.rag.retrieval.service import RetrievalFilters, ReviewRetriever


DATA_ROOT = Path(__file__).with_name("data")
BASE_MODEL = "intfloat/multilingual-e5-base"
BASE_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
BASE_DIMENSION = 768
BASE_COLLECTION = "olist_reviews_e5_base_experiment_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("index", "evaluate", "score", "all", "delete")
    )
    parser.add_argument("--environment", default="local")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-root",
        default="/workspace/reports/evaluation/comparisons/e5-base",
    )
    return parser.parse_args()


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def experiment_collection(client):
    expected = {
        "hnsw:space": "cosine",
        "collection_version": "experiment-e5-base-v1",
        "embedding_model": BASE_MODEL,
        "embedding_revision": BASE_REVISION,
        "embedding_dimension": str(BASE_DIMENSION),
    }
    collection = client.get_or_create_collection(
        name=BASE_COLLECTION,
        metadata=expected,
    )
    actual = collection.metadata or {}
    for key, value in expected.items():
        if str(actual.get(key)) != str(value):
            raise ValueError(
                "Experiment collection metadata mismatch for {}: {!r} != {!r}".format(
                    key, actual.get(key), value
                )
            )
    return collection


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def index_base(config: dict, output_root: Path, batch_size: int) -> dict:
    client = chroma_client(config)
    source = get_collection(client, config)
    target = experiment_collection(client)
    existing_ids = set()
    offset = 0
    while True:
        page = target.get(limit=1000, offset=offset, include=[])
        ids = page.get("ids") or []
        existing_ids.update(ids)
        if len(ids) < 1000:
            break
        offset += len(ids)

    embedder = MultilingualE5Embedder(
        BASE_MODEL,
        BASE_REVISION,
        passage_prefix=config["rag"]["passage_prefix"],
        query_prefix=config["rag"]["query_prefix"],
    )
    started = time.perf_counter()
    source_count = source.count()
    upserted = 0
    skipped = 0
    seen = set()
    offset = 0
    while offset < source_count:
        page = source.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = page.get("ids") or []
        documents = page.get("documents") or []
        metadatas = page.get("metadatas") or []
        if not ids:
            break
        seen.update(ids)
        pending = [index for index, document_id in enumerate(ids) if document_id not in existing_ids]
        if pending:
            pending_documents = [documents[index] for index in pending]
            target.upsert(
                ids=[ids[index] for index in pending],
                documents=pending_documents,
                metadatas=[metadatas[index] for index in pending],
                embeddings=embedder.embed_documents(pending_documents),
            )
            upserted += len(pending)
        skipped += len(ids) - len(pending)
        offset += len(ids)
        if offset % (batch_size * 10) == 0 or offset == source_count:
            print("E5_BASE_INDEX_PROGRESS={}/{}".format(offset, source_count), flush=True)

    stale = sorted(existing_ids - seen)
    for batch in chunks(stale, batch_size):
        target.delete(ids=list(batch))
    final_count = target.count()
    if final_count != source_count:
        raise ValueError(
            "Experiment collection has {} documents; expected {}".format(
                final_count, source_count
            )
        )
    result = {
        "source_collection": source.name,
        "target_collection": target.name,
        "embedding_model": BASE_MODEL,
        "embedding_revision": BASE_REVISION,
        "embedding_dimension": BASE_DIMENSION,
        "documents_expected": source_count,
        "documents_upserted": upserted,
        "documents_skipped_existing": skipped,
        "stale_documents_removed": len(stale),
        "documents_after": final_count,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output_root / "index_summary.json", result)
    print("E5_BASE_INDEX=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def build_reranker(config: dict) -> MultilingualCrossEncoderReranker:
    rag = config["rag"]
    return MultilingualCrossEncoderReranker(
        rag["reranker_model"],
        rag["reranker_revision"],
        batch_size=int(rag["reranker_batch_size"]),
        max_length=int(rag["reranker_max_length"]),
    )


def build_retriever(config: dict, variant: str) -> ReviewRetriever:
    rag = config["rag"]
    client = chroma_client(config)
    if variant == "e5-small":
        collection = get_collection(client, config)
        model = rag["embedding_model"]
        revision = rag["embedding_revision"]
    elif variant == "e5-base":
        collection = experiment_collection(client)
        model = BASE_MODEL
        revision = BASE_REVISION
    else:
        raise ValueError("Unknown embedding variant: {}".format(variant))
    return ReviewRetriever(
        collection,
        MultilingualE5Embedder(
            model,
            revision,
            passage_prefix=rag["passage_prefix"],
            query_prefix=rag["query_prefix"],
        ),
        reranker=build_reranker(config),
        candidate_k=int(rag["retrieval_candidate_k"]),
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def run_variant(config: dict, variant: str, queries: List[dict], top_k: int) -> dict:
    retriever = build_retriever(config, variant)
    first = queries[0]
    # Exclude one-time model loading from steady-state latency.
    retriever.retrieve(
        first["question_it"],
        top_k=top_k,
        filters=RetrievalFilters(**first["filters"]),
        use_reranker=True,
    )
    runs = []
    timings: Dict[str, List[float]] = {
        "semantic_only": [],
        "metadata_plus_semantic": [],
        "metadata_plus_semantic_reranked": [],
    }
    try:
        for number, query in enumerate(queries, start=1):
            methods = {
                "semantic_only": (RetrievalFilters(), False),
                "metadata_plus_semantic": (
                    RetrievalFilters(**query["filters"]),
                    False,
                ),
                "metadata_plus_semantic_reranked": (
                    RetrievalFilters(**query["filters"]),
                    True,
                ),
            }
            run = {"query": query, "methods": {}}
            for method, (filters, use_reranker) in methods.items():
                started = time.perf_counter()
                results = retriever.retrieve(
                    query["question_it"],
                    top_k=top_k,
                    filters=filters,
                    use_reranker=use_reranker,
                )
                timings[method].append(time.perf_counter() - started)
                run["methods"][method] = [
                    {
                        "review_id": item["review_id"],
                        "document_original": item["document_original"],
                        "review_score": item["metadata"].get("review_score"),
                        "product_category": item["metadata"].get("product_category"),
                        "customer_state": item["metadata"].get("customer_state"),
                        "purchase_month": item["metadata"].get("purchase_month"),
                    }
                    for item in results
                ]
            runs.append(run)
            print(
                "EMBEDDING_EVAL_PROGRESS={} {}/{}".format(
                    variant, number, len(queries)
                ),
                flush=True,
            )
    finally:
        retriever.release_models()
    latency = {
        method: {
            "mean_seconds": round(statistics.mean(values), 4),
            "median_seconds": round(statistics.median(values), 4),
            "p95_seconds": round(percentile(values, 0.95), 4),
        }
        for method, values in timings.items()
    }
    return {"variant": variant, "runs": runs, "latency": latency}


def evaluate(config: dict, output_root: Path, top_k: int) -> dict:
    queries = json.loads(
        (DATA_ROOT / "retrieval_queries.json").read_text(encoding="utf-8")
    )
    qrels = json.loads(
        (DATA_ROOT / "retrieval_qrels.json").read_text(encoding="utf-8")
    )["relevant_ids"]
    reference = json.loads(
        (DATA_ROOT / "retrieval_benchmark_pool.json").read_text(encoding="utf-8")
    )
    reference_ids = {
        run["query"]["query_id"]: {
            method: set(ids) for method, ids in run["methods"].items()
        }
        for run in reference["runs"]
    }
    judged_pool = {
        query_id: set().union(*methods.values())
        for query_id, methods in reference_ids.items()
    }
    variants = [
        run_variant(config, "e5-small", queries, top_k),
        run_variant(config, "e5-base", queries, top_k),
    ]
    annotations: Dict[tuple, dict] = {}
    summary = {}
    for variant in variants:
        metrics: Dict[str, List[dict]] = {}
        for run in variant["runs"]:
            query_id = run["query"]["query_id"]
            relevant = set(qrels[query_id])
            for method, items in run["methods"].items():
                ranked = [item["review_id"] for item in items]
                hits = len(set(ranked) & relevant)
                judged = len(set(ranked) & judged_pool[query_id])
                unjudged = top_k - judged
                overlap = len(set(ranked) & reference_ids[query_id][method])
                metrics.setdefault(method, []).append(
                    {
                        "query_id": query_id,
                        "known_relevant_at_k": hits,
                        "known_precision_lower_bound_at_k": hits / top_k,
                        "possible_precision_upper_bound_at_k": (hits + unjudged) / top_k,
                        "judgment_coverage_at_k": judged / top_k,
                        "known_positive_recall_at_k": hits / len(relevant),
                        "known_positive_hit_at_k": float(hits > 0),
                        "reference_overlap_at_k": overlap / top_k,
                    }
                )
                for rank, item in enumerate(items, start=1):
                    key = (query_id, item["review_id"])
                    record = annotations.setdefault(
                        key,
                        {
                            "query_id": query_id,
                            "question_it": run["query"]["question_it"],
                            "expected_aspects": run["query"]["expected_aspects"],
                            **item,
                            "retrieved_by": [],
                            "previously_known_relevant": item["review_id"] in relevant,
                            "previously_judged": item["review_id"] in judged_pool[query_id],
                            "relevant": (
                                item["review_id"] in relevant
                                if item["review_id"] in judged_pool[query_id]
                                else None
                            ),
                            "annotation_note": "",
                        },
                    )
                    record["retrieved_by"].append(
                        {"variant": variant["variant"], "method": method, "rank": rank}
                    )
        macro = {
            method: {
                key: round(statistics.mean(row[key] for row in rows), 4)
                for key in (
                    "known_precision_lower_bound_at_k",
                    "possible_precision_upper_bound_at_k",
                    "judgment_coverage_at_k",
                    "known_positive_recall_at_k",
                    "known_positive_hit_at_k",
                    "reference_overlap_at_k",
                )
            }
            for method, rows in metrics.items()
        }
        summary[variant["variant"]] = {
            "macro_metrics": macro,
            "latency": variant["latency"],
        }
    result = {
        "schema_version": "1.0",
        "top_k": top_k,
        "query_count": len(queries),
        "comparison_scope": "same corpus, queries, filters, candidate_k and reranker",
        "metric_warning": (
            "Qrels come from the E5-small Phase 7 pool. Known precision is only a "
            "lower bound until newly retrieved E5-base reviews are annotated."
        ),
        "models": {
            "e5-small": {
                "name": config["rag"]["embedding_model"],
                "revision": config["rag"]["embedding_revision"],
                "dimension": config["rag"]["embedding_dimension"],
                "collection": config["chroma"]["collection"],
            },
            "e5-base": {
                "name": BASE_MODEL,
                "revision": BASE_REVISION,
                "dimension": BASE_DIMENSION,
                "collection": BASE_COLLECTION,
            },
        },
        "summary": summary,
        "variants": variants,
    }
    write_json(output_root / "comparison.json", result)
    write_json(
        output_root / "annotations_template.json",
        sorted(annotations.values(), key=lambda row: (row["query_id"], row["review_id"])),
    )
    write_json(
        output_root / "new_annotations_required.json",
        sorted(
            (row for row in annotations.values() if row["relevant"] is None),
            key=lambda row: (row["query_id"], row["review_id"]),
        ),
    )
    print("EMBEDDING_COMPARISON=" + json.dumps(summary, sort_keys=True), flush=True)
    return result


def score_adjudicated(output_root: Path, top_k: int) -> dict:
    comparison = json.loads(
        (output_root / "comparison.json").read_text(encoding="utf-8")
    )
    original_qrels = json.loads(
        (DATA_ROOT / "retrieval_qrels.json").read_text(encoding="utf-8")
    )["relevant_ids"]
    supplemental = json.loads(
        (DATA_ROOT / "e5_base_reranked_qrels.json").read_text(encoding="utf-8")
    )
    variants = {variant["variant"]: variant for variant in comparison["variants"]}
    small_runs = {
        run["query"]["query_id"]: run for run in variants["e5-small"]["runs"]
    }
    base_runs = {
        run["query"]["query_id"]: run for run in variants["e5-base"]["runs"]
    }
    method = "metadata_plus_semantic_reranked"
    rows = []
    expected_new = set()
    labeled_new = set()
    for query_id, base_run in base_runs.items():
        small_ids = {
            item["review_id"] for item in small_runs[query_id]["methods"][method]
        }
        base_ids = {
            item["review_id"] for item in base_run["methods"][method]
        }
        reference_pool = set().union(
            *comparison_reference_methods(query_id)
        )
        new_ids = base_ids - reference_pool
        expected_new.update((query_id, review_id) for review_id in new_ids)
        positive_new = set(supplemental["relevant_ids"].get(query_id, []))
        negative_new = set(supplemental["non_relevant_ids"].get(query_id, []))
        labeled_new.update(
            (query_id, review_id) for review_id in positive_new | negative_new
        )
        if positive_new & negative_new:
            raise ValueError("Conflicting supplemental judgments for {}".format(query_id))
        relevant = (set(original_qrels[query_id]) & (small_ids | base_ids)) | positive_new
        for variant_name, ranked_ids in (
            (
                "e5-small",
                [
                    item["review_id"]
                    for item in small_runs[query_id]["methods"][method]
                ],
            ),
            (
                "e5-base",
                [item["review_id"] for item in base_run["methods"][method]],
            ),
        ):
            hits = len(set(ranked_ids) & relevant)
            rows.append(
                {
                    "query_id": query_id,
                    "variant": variant_name,
                    "relevant_at_k": hits,
                    "precision_at_k": hits / top_k,
                    "recall_in_two_model_pool_at_k": hits / len(relevant),
                    "hit_rate_at_k": float(hits > 0),
                }
            )
    if expected_new != labeled_new:
        missing = sorted(expected_new - labeled_new)
        extra = sorted(labeled_new - expected_new)
        raise ValueError(
            "Supplemental judgments mismatch: missing={!r}, extra={!r}".format(
                missing, extra
            )
        )
    summary = {}
    for variant_name in ("e5-small", "e5-base"):
        selected = [row for row in rows if row["variant"] == variant_name]
        summary[variant_name] = {
            key: round(statistics.mean(row[key] for row in selected), 4)
            for key in (
                "precision_at_k",
                "recall_in_two_model_pool_at_k",
                "hit_rate_at_k",
            )
        }
    result = {
        "schema_version": "1.0",
        "top_k": top_k,
        "query_count": len(base_runs),
        "method": method,
        "pooling": "union of E5-small and E5-base top-k for the compared method",
        "supplemental_annotation_method": supplemental["annotation_method"],
        "new_pairs_annotated": len(expected_new),
        "summary": summary,
        "per_query": rows,
    }
    write_json(output_root / "adjudicated_metrics.json", result)
    print("EMBEDDING_ADJUDICATED=" + json.dumps(summary, sort_keys=True), flush=True)
    return result


def comparison_reference_methods(query_id: str) -> List[set]:
    reference = json.loads(
        (DATA_ROOT / "retrieval_benchmark_pool.json").read_text(encoding="utf-8")
    )
    run = next(
        item for item in reference["runs"] if item["query"]["query_id"] == query_id
    )
    return [set(ids) for ids in run["methods"].values()]


def delete_experiment(config: dict) -> None:
    client = chroma_client(config)
    names = {item.name for item in client.list_collections()}
    if BASE_COLLECTION in names:
        client.delete_collection(BASE_COLLECTION)
        print("E5_BASE_COLLECTION_DELETED={}".format(BASE_COLLECTION))
    else:
        print("E5_BASE_COLLECTION_NOT_FOUND={}".format(BASE_COLLECTION))


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.batch_size < 1:
        raise ValueError("top-k and batch-size must be positive")
    config = load_config(args.environment)
    output_root = Path(args.output_root)
    if args.mode in ("index", "all"):
        index_base(config, output_root, args.batch_size)
    if args.mode in ("evaluate", "all"):
        evaluate(config, output_root, args.top_k)
        score_adjudicated(output_root, args.top_k)
    if args.mode == "score":
        score_adjudicated(output_root, args.top_k)
    if args.mode == "delete":
        delete_experiment(config)


if __name__ == "__main__":
    main()
