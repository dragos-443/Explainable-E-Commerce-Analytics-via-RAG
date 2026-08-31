"""Lazy multilingual cross-encoder reranking for Chroma candidates."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


class MultilingualCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive")
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                revision=self.revision,
                max_length=self.max_length,
                device="cpu",
                trust_remote_code=False,
            )
        return self._model

    def rerank(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not candidates:
            return []
        pairs = [(question, item["document_original"]) for item in candidates]
        scores = self._load().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scored = []
        for item, score in zip(candidates, scores):
            enriched = dict(item)
            enriched["reranker_score"] = float(score)
            scored.append(enriched)
        scored.sort(
            key=lambda item: (item["reranker_score"], item["similarity_score"]),
            reverse=True,
        )
        selected = scored[:top_k]
        for rank, item in enumerate(selected, start=1):
            item["reranked_rank"] = rank
        return selected

    def unload(self) -> None:
        self._model = None
