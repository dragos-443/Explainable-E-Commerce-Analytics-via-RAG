"""Semantic retrieval over original reviews with optional metadata filters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import gc
from typing import Any, Dict, List, Optional

from ecommerce_rag.rag.themes import ALL_THEMES


def month_index(value: str) -> int:
    try:
        year, month = (int(part) for part in value.split("-"))
        parsed = date(year, month, 1)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Months must use YYYY-MM format: {!r}".format(value))
    return parsed.year * 12 + parsed.month


@dataclass(frozen=True)
class RetrievalFilters:
    product_category: Optional[str] = None
    customer_state: Optional[str] = None
    start_month: Optional[str] = None
    end_month: Optional[str] = None
    max_review_score: Optional[int] = None
    theme: Optional[str] = None

    def validated(self) -> "RetrievalFilters":
        if self.end_month and not self.start_month:
            raise ValueError("end_month requires start_month")
        start = month_index(self.start_month) if self.start_month else None
        end = month_index(self.end_month) if self.end_month else start
        if start is not None and end is not None and start > end:
            raise ValueError("start_month must not be after end_month")
        if self.max_review_score is not None and not 1 <= self.max_review_score <= 5:
            raise ValueError("max_review_score must be between 1 and 5")
        if self.theme is not None and self.theme not in ALL_THEMES:
            raise ValueError("Unknown complaint theme: {}".format(self.theme))
        return RetrievalFilters(
            product_category=self.product_category or None,
            customer_state=(self.customer_state.upper() if self.customer_state else None),
            start_month=self.start_month,
            end_month=self.end_month or self.start_month,
            max_review_score=self.max_review_score,
            theme=self.theme,
        )


def build_where(filters: RetrievalFilters) -> Optional[Dict[str, Any]]:
    filters = filters.validated()
    conditions: List[Dict[str, Any]] = []
    if filters.product_category:
        conditions.append({"product_category": filters.product_category})
    if filters.customer_state:
        conditions.append({"customer_state": filters.customer_state})
    if filters.start_month:
        conditions.append(
            {"purchase_month_index": {"$gte": month_index(filters.start_month)}}
        )
        conditions.append(
            {"purchase_month_index": {"$lte": month_index(filters.end_month)}}
        )
    if filters.max_review_score is not None:
        conditions.append({"review_score": {"$lte": filters.max_review_score}})
    if filters.theme:
        conditions.append({"theme_{}".format(filters.theme): True})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class ReviewRetriever:
    def __init__(
        self,
        collection,
        embedder,
        translator=None,
        *,
        reranker=None,
        candidate_k: int = 50,
    ):
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        self.collection = collection
        self.embedder = embedder
        self.translator = translator
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: Optional[RetrievalFilters] = None,
        translate: bool = False,
        use_reranker: bool = True,
    ) -> List[Dict[str, Any]]:
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        rerank = use_reranker and self.reranker is not None
        requested_candidates = max(top_k, self.candidate_k) if rerank else top_k
        selected = filters or RetrievalFilters()
        query = self.collection.query(
            query_embeddings=[self.embedder.embed_query(question.strip())],
            n_results=requested_candidates,
            where=build_where(selected),
            include=["documents", "metadatas", "distances"],
        )
        ids = (query.get("ids") or [[]])[0]
        documents = (query.get("documents") or [[]])[0]
        metadatas = (query.get("metadatas") or [[]])[0]
        distances = (query.get("distances") or [[]])[0]
        results = []
        for semantic_rank, (document_id, document, metadata, distance) in enumerate(zip(
            ids, documents, metadatas, distances
        ), start=1):
            metadata = metadata or {}
            item = {
                "document_id": document_id,
                "review_id": metadata.get("review_id", document_id),
                "document_original": document,
                "title_original": metadata.get("original_title"),
                "message_original": metadata.get("original_message"),
                "metadata": metadata,
                "distance": distance,
                "similarity_score": 1.0 - distance,
                "semantic_rank": semantic_rank,
                "reranker_score": None,
                "reranked_rank": None,
                "translation": None,
            }
            results.append(item)
        if rerank:
            results = self.reranker.rerank(question.strip(), results, top_k)
        else:
            results = results[:top_k]
        if translate:
            for item in results:
                if self.translator is None:
                    raise ValueError("translate=True requires a translator")
                item["translation"] = self.translator.translate_review(
                    item["review_id"],
                    item["title_original"],
                    item["message_original"],
                )
        return results

    def release_models(self) -> None:
        """Release sequential retrieval models before the local LLM is loaded."""
        for component in (self.embedder, self.reranker, self.translator):
            unload = getattr(component, "unload", None)
            if unload is not None:
                unload()
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (AttributeError, OSError):
            pass
