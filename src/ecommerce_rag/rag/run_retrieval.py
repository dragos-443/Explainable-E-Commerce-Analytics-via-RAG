"""Run one Italian semantic query against the versioned Chroma collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecommerce_rag.common.config import load_config
from ecommerce_rag.rag.embeddings.model import MultilingualE5Embedder
from ecommerce_rag.rag.index import chroma_client, get_collection
from ecommerce_rag.rag.retrieval.service import RetrievalFilters, ReviewRetriever
from ecommerce_rag.rag.translation import CachedMarianTranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--environment", default="local")
    parser.add_argument("--query-id", default="retrieval-query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--product-category")
    parser.add_argument("--customer-state")
    parser.add_argument("--start-month")
    parser.add_argument("--end-month")
    parser.add_argument("--max-review-score", type=int)
    parser.add_argument("--theme")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--output-root", default="/workspace/reports/rag/phase4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.environment)
    rag = config["rag"]
    embedder = MultilingualE5Embedder(
        rag["embedding_model"],
        rag["embedding_revision"],
        passage_prefix=rag["passage_prefix"],
        query_prefix=rag["query_prefix"],
    )
    translator = None
    if args.translate:
        translator = CachedMarianTranslator(
            rag["translation_model"],
            rag["translation_revision"],
            rag["translation_cache"],
            target_prefix=rag["translation_target_prefix"],
        )
    filters = RetrievalFilters(
        product_category=args.product_category,
        customer_state=args.customer_state,
        start_month=args.start_month,
        end_month=args.end_month,
        max_review_score=args.max_review_score,
        theme=args.theme,
    ).validated()
    results = ReviewRetriever(
        get_collection(chroma_client(config), config), embedder, translator
    ).retrieve(
        args.question,
        top_k=args.top_k,
        filters=filters,
        translate=args.translate,
    )
    payload = {
        "query_id": args.query_id,
        "question_original": args.question,
        "question_language": "it",
        "filters": filters.__dict__,
        "top_k": args.top_k,
        "result_count": len(results),
        "results": results,
    }
    output = Path(args.output_root) / "{}.json".format(args.query_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE4_RETRIEVAL=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
