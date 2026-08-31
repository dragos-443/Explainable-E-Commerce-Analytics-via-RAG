"""On-demand Italian translation with a persistent, versioned SQLite cache."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class CachedMarianTranslator:
    def __init__(
        self,
        model_name: str,
        revision: str,
        cache_path: str,
        target_prefix: str = ">>ita<< ",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.cache_path = Path(cache_path)
        self.target_prefix = target_prefix
        self._tokenizer = None
        self._model = None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_cache()

    @property
    def version(self) -> str:
        return "{}@{}".format(self.model_name, self.revision)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.cache_path))

    def _initialize_cache(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    review_id TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    translator_version TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    title_translation TEXT,
                    message_translation TEXT,
                    PRIMARY KEY (
                        review_id, target_language, translator_version, source_hash
                    )
                )
                """
            )

    @staticmethod
    def _source_hash(title: Optional[str], message: Optional[str]) -> str:
        payload = "{}\0{}".format(title or "", message or "")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cached(
        self, review_id: str, source_hash: str, target_language: str
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT title_translation, message_translation
                FROM translations
                WHERE review_id = ? AND target_language = ?
                  AND translator_version = ? AND source_hash = ?
                """,
                (review_id, target_language, self.version, source_hash),
            ).fetchone()
        if row is None:
            return None
        return {
            "title": row[0],
            "message": row[1],
            "target_language": target_language,
            "translator_version": self.version,
            "automatic": True,
            "cache_hit": True,
            "status": "translated",
            "error": None,
        }

    def _load(self) -> None:
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.revision, trust_remote_code=False
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=False,
            )
            self._model.eval()

    def _translate_texts(self, texts: Sequence[str]) -> List[str]:
        import torch

        self._load()
        prepared = [self.target_prefix + text for text in texts]
        encoded = self._tokenizer(
            prepared,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded, max_new_tokens=512, num_beams=4, do_sample=False
            )
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

    def translate_review(
        self,
        review_id: str,
        title: Optional[str],
        message: Optional[str],
        target_language: str = "it",
    ) -> Dict[str, Any]:
        source_hash = self._source_hash(title, message)
        cached = self._cached(review_id, source_hash, target_language)
        if cached is not None:
            return cached
        present = [("title", title), ("message", message)]
        present = [(field, value) for field, value in present if value]
        try:
            translated_values = self._translate_texts([value for _, value in present])
            translated = {field: value for (field, _), value in zip(present, translated_values)}
            title_translation = translated.get("title")
            message_translation = translated.get("message")
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO translations VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_id,
                        target_language,
                        self.version,
                        source_hash,
                        title_translation,
                        message_translation,
                    ),
                )
            return {
                "title": title_translation,
                "message": message_translation,
                "target_language": target_language,
                "translator_version": self.version,
                "automatic": True,
                "cache_hit": False,
                "status": "translated",
                "error": None,
            }
        except Exception as error:
            return {
                "title": None,
                "message": None,
                "target_language": target_language,
                "translator_version": self.version,
                "automatic": True,
                "cache_hit": False,
                "status": "failed_original_available",
                "error": "{}: {}".format(type(error).__name__, error),
            }

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
