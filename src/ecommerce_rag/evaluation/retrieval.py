"""Collect pooled judgments and score baseline/proposed Phase 7 retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from ecommerce_rag.common.config import load_config
from ecommerce_rag.evaluation.metrics import macro_average, retrieval_metrics
from ecommerce_rag.rag.retrieval.factory import build_review_retriever
from ecommerce_rag.rag.retrieval.service import RetrievalFilters, ReviewRetriever


DATA_ROOT = Path(__file__).with_name("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "score"))
    parser.add_argument("--environment", default="local")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-root", default="/workspace/reports/evaluation/phase7")
    return parser.parse_args()


def load_queries() -> List[dict]:
    return json.loads((DATA_ROOT / "retrieval_queries.json").read_text(encoding="utf-8"))


def collect(environment: str, top_k: int, output_root: Path) -> None:
    config = load_config(environment)
    rag = config["rag"]
    retriever = build_review_retriever(config)
    runs = []
    annotations = []
    for query in load_queries():
        methods = {
            "semantic_only": (RetrievalFilters(), False),
            "metadata_plus_semantic": (
                RetrievalFilters(**query["filters"]), False
            ),
            "metadata_plus_semantic_reranked": (
                RetrievalFilters(**query["filters"]), True
            ),
        }
        pooled: Dict[str, dict] = {}
        run = {"query": query, "methods": {}}
        for method, (filters, use_reranker) in methods.items():
            results = retriever.retrieve(
                query["question_it"],
                top_k=top_k,
                filters=filters,
                use_reranker=use_reranker,
            )
            run["methods"][method] = [item["review_id"] for item in results]
            for rank, item in enumerate(results, start=1):
                record = pooled.setdefault(
                    item["review_id"],
                    {
                        "review_id": item["review_id"],
                        "document_original": item["document_original"],
                        "review_score": item["metadata"].get("review_score"),
                        "product_category": item["metadata"].get("product_category"),
                        "customer_state": item["metadata"].get("customer_state"),
                        "purchase_month": item["metadata"].get("purchase_month"),
                        "automatic_themes": item["metadata"].get("themes_csv", "").split(","),
                        "retrieved_by": [],
                    },
                )
                record["retrieved_by"].append({"method": method, "rank": rank})
        runs.append(run)
        for item in sorted(pooled.values(), key=lambda value: value["review_id"]):
            annotations.append(
                {
                    "query_id": query["query_id"],
                    "question_it": query["question_it"],
                    "expected_aspects": query["expected_aspects"],
                    **item,
                    "relevant": None,
                    "annotation_note": "",
                }
            )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "top_k": top_k,
        "query_count": len(runs),
        "query_language": "it",
        "evidence_language": "pt",
        "recall_definition": "recall within the complete judged pool of all top-k runs",
        "candidate_k": retriever.candidate_k,
        "reranker": {
            "model": rag["reranker_model"],
            "revision": rag["reranker_revision"],
        },
        "runs": runs,
    }
    (output_root / "retrieval_pool.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "retrieval_annotations_template.json").write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PHASE7_RETRIEVAL_POOL="
        + json.dumps(
            {"query_count": len(runs), "judgments_required": len(annotations)},
            sort_keys=True,
        )
    )


def score(top_k: int, output_root: Path) -> None:
    pool = json.loads((output_root / "retrieval_pool.json").read_text(encoding="utf-8"))
    qrels = json.loads((DATA_ROOT / "retrieval_qrels.json").read_text(encoding="utf-8"))
    judged = {
        query_id: set(document_ids)
        for query_id, document_ids in qrels["relevant_ids"].items()
    }
    rows = []
    for run in pool["runs"]:
        query_id = run["query"]["query_id"]
        pool_ids = set().union(*[set(ids) for ids in run["methods"].values()])
        if query_id not in judged or not judged[query_id].issubset(pool_ids):
            raise ValueError("Qrels contain missing or unknown ids for {}".format(query_id))
        for method, ranked_ids in run["methods"].items():
            rows.append(
                {
                    "query_id": query_id,
                    "method": method,
                    **retrieval_metrics(ranked_ids, judged[query_id], top_k),
                    "relevant_in_pool": len(judged[query_id]),
                }
            )
    by_method = {}
    for method in (
        "semantic_only",
        "metadata_plus_semantic",
        "metadata_plus_semantic_reranked",
    ):
        selected = [row for row in rows if row["method"] == method]
        by_method[method] = macro_average(
            {
                key: row[key]
                for key in ("precision_at_k", "recall_at_k", "hit_rate_at_k")
            }
            for row in selected
        )
    result = {
        "schema_version": "1.0",
        "top_k": top_k,
        "query_count": pool["query_count"],
        "query_language": pool["query_language"],
        "evidence_language": pool["evidence_language"],
        "candidate_k": pool["candidate_k"],
        "reranker": pool["reranker"],
        "recall_definition": pool["recall_definition"],
        "annotation_protocol": qrels["annotation_protocol"],
        "per_query": rows,
        "macro_metrics": by_method,
    }
    (output_root / "retrieval_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE7_RETRIEVAL_METRICS=" + json.dumps(by_method, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    output_root = Path(args.output_root)
    if args.mode == "collect":
        collect(args.environment, args.top_k, output_root)
    else:
        score(args.top_k, output_root)


if __name__ == "__main__":
    main()
