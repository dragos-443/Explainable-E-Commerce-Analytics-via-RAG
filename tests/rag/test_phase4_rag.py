"""Unit tests for Phase 4 document, index, retrieval and translation rules."""

from __future__ import annotations

from datetime import datetime

import pytest
from pyspark.sql import Row, SparkSession, functions as F

from ecommerce_rag.rag.documents import build_rag_documents
from ecommerce_rag.rag.index import row_metadata, synchronize_collection
from ecommerce_rag.rag.retrieval.service import (
    RetrievalFilters,
    ReviewRetriever,
    build_where,
)
from ecommerce_rag.rag.themes import classify_review_themes
from ecommerce_rag.rag.translation import CachedMarianTranslator


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[1]")
        .appName("phase-4-rag-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_theme_classifier_covers_every_eligible_review(spark: SparkSession) -> None:
    reviews = spark.createDataFrame(
        [
            ("a", 1, "Ainda não recebi e quero reembolso", True),
            ("b", 5, "Produto excelente e entrega perfeita", True),
            ("c", 2, "ru", True),
            ("d", 1, None, False),
        ],
        ["review_id", "review_score", "review_text", "text_is_eligible"],
    )
    assignments = classify_review_themes(reviews, "test-v1")
    grouped = {}
    for row in assignments.collect():
        grouped.setdefault(row.review_id, set()).add(row.theme)

    assert grouped["a"] == {"non_delivery", "service_or_refund"}
    assert grouped["b"] == {"other"}
    assert grouped["c"] == {"uncertain"}
    assert "d" not in grouped


def test_delivery_delay_ignores_on_time_wording(spark: SparkSession) -> None:
    reviews = spark.createDataFrame(
        [
            ("on-time", 1, "Recebi o produto no prazo mas veio com defeito", True),
            ("late", 1, "O prazo já passou e ainda não recebi", True),
        ],
        ["review_id", "review_score", "review_text", "text_is_eligible"],
    )
    grouped = {}
    for row in classify_review_themes(reviews, "test-v2").collect():
        grouped.setdefault(row.review_id, set()).add(row.theme)

    assert grouped["on-time"] == {"damaged_or_defective"}
    assert grouped["late"] == {"delivery_delay", "non_delivery"}


def test_documents_preserve_title_only_and_ambiguous_metadata(
    spark: SparkSession,
) -> None:
    reviews = spark.createDataFrame(
        [
            (
                "r1", 1, "Só título", None, datetime(2018, 3, 2),
                datetime(2018, 3, 3), "Só título", True, False, True, "pt",
                1, ["o1"], "books", "SP", "delivered", 2018, 3,
            ),
            (
                "r2", 2, None, "Não chegou", datetime(2018, 3, 2),
                datetime(2018, 3, 3), "Não chegou", False, True, True, "pt",
                2, ["o2", "o3"], None, "RJ", "delivered", 2018, 3,
            ),
        ],
        "review_id string, review_score int, review_comment_title_original string, "
        "review_comment_message_original string, review_creation_date timestamp, "
        "review_answer_timestamp timestamp, review_text string, has_title boolean, "
        "has_message boolean, text_is_eligible boolean, review_language string, "
        "linked_order_count long, linked_order_ids array<string>, product_category string, "
        "customer_state string, order_status string, purchase_year int, purchase_month int",
    )
    orders = spark.createDataFrame(
        [
            ("o1", ["p1"], 1, True, "books", 2.0, True),
            ("o2", ["p2"], 1, True, "toys", 1.0, True),
            ("o3", ["p3"], 1, True, "books", 1.0, True),
        ],
        "order_id string, product_ids array<string>, item_count long, "
        "is_strict_mono_category boolean, order_category string, "
        "delivery_delay_days double, is_late boolean",
    )
    links = spark.createDataFrame(
        [("r1", "o1"), ("r2", "o2"), ("r2", "o3")],
        ["review_id", "order_id"],
    )
    themes = classify_review_themes(reviews, "test-v1")
    rows = {
        row.review_id: row
        for row in build_rag_documents(reviews, orders, links, themes).collect()
    }

    assert rows["r1"].document_text == "Title: Só título"
    assert rows["r1"].text_source == "title"
    assert rows["r1"].order_id == "o1"
    assert rows["r1"].product_id == "p1"
    assert rows["r2"].document_text == "Comment: Não chegou"
    assert rows["r2"].order_id is None
    assert rows["r2"].product_id is None
    assert rows["r2"].is_single_category_order is False


class FakeEmbedder:
    def __init__(self):
        self.document_calls = 0
        self.last_query = None

    def embed_documents(self, texts):
        self.document_calls += 1
        return [[float(len(text)), 0.0] for text in texts]

    def embed_query(self, text):
        self.last_query = text
        return [1.0, 0.0]


class FakeCollection:
    def __init__(self):
        self.records = {}
        self.last_where = None

    def get(self, limit, offset, include):
        ids = sorted(self.records)[offset : offset + limit]
        return {"ids": ids, "metadatas": [self.records[i][1] for i in ids]}

    def upsert(self, ids, documents, metadatas, embeddings):
        for values in zip(ids, documents, metadatas, embeddings):
            self.records[values[0]] = (values[1], values[2], values[3])

    def update(self, ids, metadatas):
        for document_id, metadata in zip(ids, metadatas):
            document, _, embedding = self.records[document_id]
            self.records[document_id] = (document, metadata, embedding)

    def delete(self, ids):
        for document_id in ids:
            self.records.pop(document_id, None)

    def count(self):
        return len(self.records)

    def query(self, query_embeddings, n_results, where, include):
        self.last_where = where
        return {
            "ids": [["r1"]],
            "documents": [["Comment: Não chegou"]],
            "metadatas": [[{"review_id": "r1", "original_message": "Não chegou"}]],
            "distances": [[0.1]],
        }


def test_upsert_is_idempotent_and_metadata_has_theme_flags(
    spark: SparkSession,
) -> None:
    documents = spark.createDataFrame(
        [
            ("r1", "h1", "Title: ótimo", "r1", "ótimo", None, ["other"], 5)
        ],
        "document_id string, document_hash string, document_text string, "
        "review_id string, review_comment_title_original string, "
        "review_comment_message_original string, themes array<string>, review_score int",
    )
    collection = FakeCollection()
    embedder = FakeEmbedder()

    first = synchronize_collection(documents, collection, embedder, 10)
    second = synchronize_collection(documents, collection, embedder, 10)
    changed_metadata = documents.withColumn(
        "themes", F.array(F.lit("quality_or_expectation"))
    )
    metadata_change = synchronize_collection(
        changed_metadata, collection, embedder, 10
    )
    fourth = synchronize_collection(changed_metadata, collection, embedder, 10)

    assert first["documents_upserted"] == 1
    assert second["documents_upserted"] == 0
    assert second["documents_skipped_unchanged"] == 1
    assert metadata_change["documents_upserted"] == 0
    assert metadata_change["documents_metadata_updated"] == 1
    assert fourth["documents_skipped_unchanged"] == 1
    assert collection.records["r1"][1]["theme_other"] is False
    assert collection.records["r1"][1]["theme_quality_or_expectation"] is True
    assert collection.records["r1"][1]["original_title"] == "ótimo"
    assert embedder.document_calls == 1


class FakeTranslator:
    def translate_review(self, review_id, title, message):
        return {"message": "Non è arrivato", "status": "translated"}


def test_retrieval_combines_metadata_filters_and_translation() -> None:
    filters = RetrievalFilters(
        product_category="books", customer_state="sp", start_month="2018-03",
        max_review_score=2, theme="non_delivery",
    ).validated()
    where = build_where(filters)
    assert where["$and"][-1] == {"theme_non_delivery": True}
    collection = FakeCollection()
    embedder = FakeEmbedder()
    result = ReviewRetriever(collection, embedder, FakeTranslator()).retrieve(
        "Il prodotto non è arrivato", 5, filters, translate=True
    )

    assert embedder.last_query == "Il prodotto non è arrivato"
    assert collection.last_where == where
    assert result[0]["document_original"] == "Comment: Não chegou"
    assert result[0]["translation"]["message"] == "Non è arrivato"
    assert result[0]["similarity_score"] == pytest.approx(0.9)


def test_translation_cache_preserves_missing_fields(tmp_path) -> None:
    class StubTranslator(CachedMarianTranslator):
        calls = 0

        def _translate_texts(self, texts):
            self.calls += 1
            return ["traduzione:" + text for text in texts]

    translator = StubTranslator("model", "revision", str(tmp_path / "cache.sqlite"))
    first = translator.translate_review("r1", None, "Não chegou")
    second = translator.translate_review("r1", None, "Não chegou")

    assert first["title"] is None
    assert first["message"] == "traduzione:Não chegou"
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert translator.calls == 1
