"""On-demand Italian translation with a persistent, versioned SQLite cache."""

from __future__ import annotations

import hashlib
import html
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


TRANSLATION_GLOSSARY_VERSION = "pt-it-ecommerce-v1"


def apply_translation_glossary(
    source: Optional[str], translation: Optional[str]
) -> tuple:
    """Correct known domain terms only when their source expression is present."""
    if not source or not translation:
        return translation, []
    corrections = []
    corrected = translation
    if re.search(r"\bcobre(?:-|\s+)leito\b", source, flags=re.IGNORECASE):
        pattern = re.compile(r"\brame(?:-|\s+)letto\b", flags=re.IGNORECASE)

        def replace_cobre_leito(match):
            return "Copriletto" if match.group(0)[:1].isupper() else "copriletto"

        corrected, count = pattern.subn(replace_cobre_leito, corrected)
        if count:
            corrections.append(
                {
                    "rule": "cobre_leito",
                    "source_term": "cobre-leito / cobre leito",
                    "replacement_it": "copriletto",
                }
            )
    return corrected, corrections


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

    @staticmethod
    def _apply_glossary(
        result: Dict[str, Any],
        title: Optional[str],
        message: Optional[str],
        target_language: str,
    ) -> Dict[str, Any]:
        processed = dict(result)
        corrections = []
        if target_language == "it":
            processed["title"], title_corrections = apply_translation_glossary(
                title, processed.get("title")
            )
            processed["message"], message_corrections = apply_translation_glossary(
                message, processed.get("message")
            )
            corrections.extend(title_corrections)
            corrections.extend(message_corrections)
        processed["translation_glossary_version"] = TRANSLATION_GLOSSARY_VERSION
        processed["glossary_corrections"] = corrections
        return processed

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
            return self._apply_glossary(cached, title, message, target_language)
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
            result = {
                "title": title_translation,
                "message": message_translation,
                "target_language": target_language,
                "translator_version": self.version,
                "automatic": True,
                "cache_hit": False,
                "status": "translated",
                "error": None,
            }
            return self._apply_glossary(result, title, message, target_language)
        except Exception as error:
            result = {
                "title": None,
                "message": None,
                "target_language": target_language,
                "translator_version": self.version,
                "automatic": True,
                "cache_hit": False,
                "status": "failed_original_available",
                "error": "{}: {}".format(type(error).__name__, error),
            }
            return self._apply_glossary(result, title, message, target_language)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None


class CachedGoogleTranslator(CachedMarianTranslator):
    """Google Cloud Translation Basic v2 backed by the shared SQLite cache."""

    def __init__(
        self,
        api_key: str,
        cache_path: str,
        source_language: str = "pt",
        target_language: str = "it",
        timeout_seconds: int = 30,
        session=None,
    ) -> None:
        if not api_key:
            raise ValueError("GOOGLE_TRANSLATE_API_KEY is required for Google translation")
        super().__init__(
            "google-cloud-translation",
            "basic-v2-nmt",
            cache_path,
            target_prefix="",
        )
        self.api_key = api_key
        self.source_language = source_language
        self.target_language = target_language
        self.timeout_seconds = timeout_seconds
        self._session = session

    def _translate_texts(self, texts: Sequence[str]) -> List[str]:
        import requests

        session = self._session or requests
        try:
            response = session.post(
                "https://translation.googleapis.com/language/translate/v2",
                headers={"X-Goog-Api-Key": self.api_key},
                json={
                    "q": list(texts),
                    "source": self.source_language,
                    "target": self.target_language,
                    "format": "text",
                    "model": "nmt",
                },
                timeout=self.timeout_seconds,
            )
            if not response.ok:
                raise RuntimeError("HTTP {}".format(response.status_code))
            translations = response.json()["data"]["translations"]
            if len(translations) != len(texts):
                raise RuntimeError("unexpected number of translations")
            return [html.unescape(item["translatedText"]) for item in translations]
        except Exception as error:
            # Never propagate a request object or URL that could expose the API key.
            raise RuntimeError(
                "Google Cloud Translation request failed: {}".format(error)
            ) from None

    def unload(self) -> None:
        return None


class FallbackTranslator:
    """Use the local translator only when the configured cloud provider fails."""

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback

    def translate_review(self, review_id, title, message, target_language="it"):
        primary_result = self.primary.translate_review(
            review_id, title, message, target_language
        )
        if primary_result.get("status") == "translated":
            primary_result["translation_provider"] = "google"
            return primary_result
        fallback_result = self.fallback.translate_review(
            review_id, title, message, target_language
        )
        fallback_result["translation_provider"] = "local_fallback"
        fallback_result["primary_error"] = primary_result.get("error")
        return fallback_result

    def unload(self) -> None:
        self.primary.unload()
        self.fallback.unload()


def build_translator(config: Dict[str, Any]):
    """Build the selected provider without storing credentials in configuration files."""
    rag = config["rag"]
    local = CachedMarianTranslator(
        rag["translation_model"],
        rag["translation_revision"],
        rag["translation_cache"],
        target_prefix=rag["translation_target_prefix"],
    )
    provider = os.getenv(
        "TRANSLATION_PROVIDER", rag.get("translation_provider", "local")
    ).strip().lower()
    if provider == "local":
        return local
    if provider != "google":
        raise ValueError("Unsupported TRANSLATION_PROVIDER: {}".format(provider))
    google = CachedGoogleTranslator(
        os.getenv("GOOGLE_TRANSLATE_API_KEY", ""),
        rag["translation_cache"],
        source_language=config["application"]["source_review_language"],
        target_language=config["application"]["language"],
    )
    return FallbackTranslator(google, local)
