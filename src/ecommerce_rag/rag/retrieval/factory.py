"""Build the configured review retriever without hardcoded model settings."""

from __future__ import annotations

from ecommerce_rag.rag.embeddings.model import MultilingualE5Embedder
from ecommerce_rag.rag.index import chroma_client, get_collection
from ecommerce_rag.rag.retrieval.reranking import MultilingualCrossEncoderReranker
from ecommerce_rag.rag.retrieval.service import ReviewRetriever


def build_review_retriever(config: dict, translator=None) -> ReviewRetriever:
    rag = config["rag"]
    embedder = MultilingualE5Embedder(
        rag["embedding_model"],
        rag["embedding_revision"],
        passage_prefix=rag["passage_prefix"],
        query_prefix=rag["query_prefix"],
    )
    reranker = MultilingualCrossEncoderReranker(
        rag["reranker_model"],
        rag["reranker_revision"],
        batch_size=int(rag["reranker_batch_size"]),
        max_length=int(rag["reranker_max_length"]),
    )
    return ReviewRetriever(
        get_collection(chroma_client(config), config),
        embedder,
        translator,
        reranker=reranker,
        candidate_k=int(rag["retrieval_candidate_k"]),
    )
