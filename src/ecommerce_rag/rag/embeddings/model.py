"""Pinned multilingual E5 embedding adapter."""

from __future__ import annotations

from typing import List, Sequence


class MultilingualE5Embedder:
    def __init__(
        self,
        model_name: str,
        revision: str,
        passage_prefix: str = "passage: ",
        query_prefix: str = "query: ",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.passage_prefix = passage_prefix
        self.query_prefix = query_prefix
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                revision=self.revision,
                device="cpu",
                trust_remote_code=False,
            )
        return self._model

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._load().encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._encode([self.passage_prefix + text for text in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._encode([self.query_prefix + text])[0]
