"""Exercise Chroma with small deterministic embeddings and an idempotent upsert."""

from __future__ import annotations

import hashlib
import math
import os
import re
import time

import chromadb

from ecommerce_rag.common.config import load_config


DIMENSIONS = 128


def embed(text: str) -> list[float]:
    """Create a stable normalized token-hashing vector for the smoke test only."""
    vector = [0.0] * DIMENSIONS
    tokens = re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % DIMENSIONS
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def connect() -> chromadb.HttpClient:
    config = load_config()
    host = os.getenv("CHROMA_HOST", config["chroma"]["host"])
    port = int(os.getenv("CHROMA_PORT", config["chroma"]["port"]))
    last_error: Exception | None = None
    for _ in range(30):
        try:
            client = chromadb.HttpClient(host=host, port=port)
            client.heartbeat()
            return client
        except Exception as error:  # Chroma may still be starting.
            last_error = error
            time.sleep(2)
    raise RuntimeError("Chroma did not become ready") from last_error


def main() -> None:
    client = connect()
    collection = client.get_or_create_collection(
        name="phase0_smoke",
        metadata={"hnsw:space": "cosine"},
    )

    ids = ["review-late", "review-product", "review-positive"]
    documents = [
        "A entrega chegou atrasada e demorou muitos dias",
        "O produto chegou quebrado e com defeito",
        "Entrega rápida e produto excelente",
    ]
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=[embed(document) for document in documents],
        metadatas=[{"kind": "smoke"}] * len(documents),
    )

    query = "consegna atrasada"
    result = collection.query(
        query_embeddings=[embed(query)],
        n_results=1,
        include=["documents", "distances"],
    )
    retrieved_id = result["ids"][0][0]
    if retrieved_id != "review-late":
        raise RuntimeError(f"Unexpected Chroma result: {retrieved_id}")

    print(
        "SMOKE_CHROMA_RESULT="
        f"query={query!r} id={retrieved_id!r} "
        f"document={result['documents'][0][0]!r}"
    )


if __name__ == "__main__":
    main()
