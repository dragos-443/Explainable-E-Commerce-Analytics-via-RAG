"""Idempotent synchronization of canonical review documents into Chroma."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
from pyspark.sql import Row, SparkSession

from ecommerce_rag.common.config import load_config
from ecommerce_rag.rag.embeddings.model import MultilingualE5Embedder
from ecommerce_rag.rag.themes import ALL_THEMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="local")
    parser.add_argument(
        "--summary-output",
        default="/workspace/reports/rag/phase4/index_summary.json",
    )
    return parser.parse_args()


def chroma_client(config: dict):
    return chromadb.HttpClient(
        host=config["chroma"]["host"], port=config["chroma"]["port"]
    )


def get_collection(client, config: dict):
    rag = config["rag"]
    expected_metadata = {
        "hnsw:space": rag["distance_metric"],
        "collection_version": rag["collection_version"],
        "embedding_model": rag["embedding_model"],
        "embedding_revision": rag["embedding_revision"],
        "embedding_dimension": str(rag["embedding_dimension"]),
    }
    collection = client.get_or_create_collection(
        name=config["chroma"]["collection"], metadata=expected_metadata
    )
    actual = collection.metadata or {}
    for key, value in expected_metadata.items():
        if str(actual.get(key)) != str(value):
            raise ValueError(
                "Chroma collection metadata mismatch for {}: {!r} != {!r}".format(
                    key, actual.get(key), value
                )
            )
    return collection


def _native(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def row_metadata(row: Row) -> Dict[str, Any]:
    values = row.asDict(recursive=True)
    themes = set(values.pop("themes") or [])
    original_title = values.pop("review_comment_title_original", None)
    original_message = values.pop("review_comment_message_original", None)
    excluded = {
        "document_text",
        "review_comment_title_original",
        "review_comment_message_original",
        "order_ids",
    }
    metadata = {
        key: _native(value)
        for key, value in values.items()
        if key not in excluded and value is not None
    }
    metadata["themes_csv"] = ",".join(sorted(themes))
    if original_title is not None:
        metadata["original_title"] = original_title
    if original_message is not None:
        metadata["original_message"] = original_message
    for theme in ALL_THEMES:
        metadata["theme_{}".format(theme)] = theme in themes
    return metadata


def metadata_fingerprint(metadata: Dict[str, Any]) -> str:
    """Hash the complete Chroma metadata payload using stable JSON."""
    payload = {
        key: value for key, value in metadata.items() if key != "metadata_hash"
    }
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def existing_fingerprints(
    collection, page_size: int = 1000
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    result: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    offset = 0
    while True:
        page = collection.get(
            limit=page_size, offset=offset, include=["metadatas"]
        )
        ids = page.get("ids") or []
        metadatas = page.get("metadatas") or []
        for document_id, metadata in zip(ids, metadatas):
            current = metadata or {}
            result[document_id] = (
                current.get("document_hash"),
                current.get("metadata_hash"),
            )
        if len(ids) < page_size:
            break
        offset += len(ids)
    return result


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def synchronize_collection(
    documents,
    collection,
    embedder: MultilingualE5Embedder,
    batch_size: int,
) -> dict:
    started = time.perf_counter()
    existing = existing_fingerprints(collection)
    seen = set()
    pending_ids: List[str] = []
    pending_texts: List[str] = []
    pending_metadata: List[Dict[str, Any]] = []
    pending_metadata_ids: List[str] = []
    pending_metadata_only: List[Dict[str, Any]] = []
    upserted = 0
    metadata_updated = 0
    skipped = 0

    def flush() -> None:
        nonlocal upserted
        if not pending_ids:
            return
        embeddings = embedder.embed_documents(pending_texts)
        collection.upsert(
            ids=list(pending_ids),
            documents=list(pending_texts),
            metadatas=list(pending_metadata),
            embeddings=embeddings,
        )
        upserted += len(pending_ids)
        pending_ids.clear()
        pending_texts.clear()
        pending_metadata.clear()

    def flush_metadata() -> None:
        nonlocal metadata_updated
        if not pending_metadata_ids:
            return
        collection.update(
            ids=list(pending_metadata_ids),
            metadatas=list(pending_metadata_only),
        )
        metadata_updated += len(pending_metadata_ids)
        pending_metadata_ids.clear()
        pending_metadata_only.clear()

    for row in documents.orderBy("document_id").toLocalIterator():
        document_id = row.document_id
        if document_id in seen:
            raise ValueError("Duplicate document_id during Chroma synchronization")
        seen.add(document_id)
        metadata = row_metadata(row)
        fingerprint = metadata_fingerprint(metadata)
        metadata["metadata_hash"] = fingerprint
        existing_document_hash, existing_metadata_hash = existing.get(
            document_id, (None, None)
        )
        if (
            existing_document_hash == row.document_hash
            and existing_metadata_hash == fingerprint
        ):
            skipped += 1
            continue
        if existing_document_hash == row.document_hash:
            pending_metadata_ids.append(document_id)
            pending_metadata_only.append(metadata)
            if len(pending_metadata_ids) >= batch_size:
                flush_metadata()
            continue
        pending_ids.append(document_id)
        pending_texts.append(row.document_text)
        pending_metadata.append(metadata)
        if len(pending_ids) >= batch_size:
            flush()
    flush()
    flush_metadata()

    stale = sorted(set(existing) - seen)
    for ids in _chunks(stale, batch_size):
        collection.delete(ids=list(ids))
    final_count = collection.count()
    if final_count != len(seen):
        raise ValueError(
            "Chroma count {} does not match {} canonical documents".format(
                final_count, len(seen)
            )
        )
    return {
        "documents_expected": len(seen),
        "documents_existing_before": len(existing),
        "documents_upserted": upserted,
        "documents_metadata_updated": metadata_updated,
        "documents_skipped_unchanged": skipped,
        "stale_documents_removed": len(stale),
        "documents_after": final_count,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.environment)
    spark = (
        SparkSession.builder.appName("phase-4-chroma-index")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        curated = "{}/olist".format(config["storage"]["curated_uri"])
        documents = spark.read.parquet("{}/rag_documents".format(curated))
        embedder = MultilingualE5Embedder(
            config["rag"]["embedding_model"],
            config["rag"]["embedding_revision"],
            passage_prefix=config["rag"]["passage_prefix"],
            query_prefix=config["rag"]["query_prefix"],
        )
        collection = get_collection(chroma_client(config), config)
        summary = synchronize_collection(
            documents,
            collection,
            embedder,
            int(config["rag"]["embedding_batch_size"]),
        )
        summary.update(
            {
                "collection": config["chroma"]["collection"],
                "embedding_model": config["rag"]["embedding_model"],
                "embedding_revision": config["rag"]["embedding_revision"],
            }
        )
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("PHASE4_INDEX=" + json.dumps(summary, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
