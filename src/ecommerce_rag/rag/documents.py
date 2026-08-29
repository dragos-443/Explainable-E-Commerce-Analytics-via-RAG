"""Build canonical RAG documents and unambiguous order metadata."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_document_text() -> F.Column:
    title = F.trim(F.col("review_comment_title_original"))
    message = F.trim(F.col("review_comment_message_original"))
    return (
        F.when(
            F.col("has_title") & F.col("has_message"),
            F.concat(F.lit("Title: "), title, F.lit("\nComment: "), message),
        )
        .when(F.col("has_title"), F.concat(F.lit("Title: "), title))
        .when(F.col("has_message"), F.concat(F.lit("Comment: "), message))
    )


def _order_metadata(orders: DataFrame, links: DataFrame) -> DataFrame:
    linked = links.join(
        orders.select(
            "order_id",
            "product_ids",
            "item_count",
            "is_strict_mono_category",
            "order_category",
            "delivery_delay_days",
            "is_late",
        ),
        "order_id",
        "left",
    )
    grouped = linked.groupBy("review_id").agg(
        F.countDistinct("order_id").alias("order_count_from_links"),
        F.sort_array(F.collect_set("order_id")).alias("order_ids_from_links"),
        F.array_distinct(F.flatten(F.collect_list("product_ids"))).alias(
            "all_product_ids"
        ),
        F.first("item_count", ignorenulls=True).alias("candidate_item_count"),
        F.min(F.col("is_strict_mono_category").cast("int")).alias(
            "all_strict_mono_category"
        ),
        F.count("order_category").alias("known_category_count"),
        F.countDistinct("order_category").alias("distinct_category_count"),
        F.count("delivery_delay_days").alias("known_delay_count"),
        F.countDistinct("delivery_delay_days").alias("distinct_delay_count"),
        F.first("delivery_delay_days", ignorenulls=True).alias("candidate_delay"),
        F.count("is_late").alias("known_late_count"),
        F.countDistinct(F.col("is_late").cast("int")).alias(
            "distinct_late_count"
        ),
        F.first("is_late", ignorenulls=True).alias("candidate_is_late"),
    )
    return (
        grouped.withColumn(
            "order_id",
            F.when(
                F.col("order_count_from_links") == 1,
                F.element_at("order_ids_from_links", 1),
            ),
        )
        .withColumn("distinct_product_count", F.size("all_product_ids"))
        .withColumn(
            "product_id",
            F.when(
                F.col("distinct_product_count") == 1,
                F.element_at("all_product_ids", 1),
            ),
        )
        .withColumn(
            "item_count",
            F.when(
                F.col("order_count_from_links") == 1,
                F.col("candidate_item_count"),
            ),
        )
        .withColumn(
            "is_single_product_order",
            (F.col("order_count_from_links") == 1)
            & (F.col("distinct_product_count") == 1),
        )
        .withColumn(
            "is_single_category_order",
            (F.col("all_strict_mono_category") == 1)
            & (F.col("known_category_count") == F.col("order_count_from_links"))
            & (F.col("distinct_category_count") == 1),
        )
        .withColumn(
            "delivery_delay_days",
            F.when(
                (F.col("known_delay_count") == F.col("order_count_from_links"))
                & (F.col("distinct_delay_count") == 1),
                F.col("candidate_delay"),
            ),
        )
        .withColumn(
            "is_late",
            F.when(
                (F.col("known_late_count") == F.col("order_count_from_links"))
                & (F.col("distinct_late_count") == 1),
                F.col("candidate_is_late"),
            ),
        )
        .select(
            "review_id",
            "order_id",
            F.col("order_count_from_links").alias("order_count"),
            F.col("order_ids_from_links").alias("order_ids"),
            "product_id",
            "distinct_product_count",
            "item_count",
            "is_single_product_order",
            "is_single_category_order",
            "delivery_delay_days",
            "is_late",
        )
    )


def build_rag_documents(
    reviews: DataFrame,
    orders: DataFrame,
    links: DataFrame,
    review_themes: DataFrame,
) -> DataFrame:
    themes = review_themes.groupBy("review_id").agg(
        F.sort_array(F.collect_set("theme")).alias("themes"),
        F.first("classifier_version").alias("theme_classifier_version"),
        F.first("classification_method").alias("theme_classification_method"),
    )
    order_metadata = _order_metadata(orders, links)
    documents = (
        reviews.where(F.col("text_is_eligible"))
        .join(order_metadata, "review_id", "left")
        .join(themes, "review_id", "left")
        .withColumn("document_id", F.col("review_id"))
        .withColumn("document_text", build_document_text())
        .withColumn(
            "text_source",
            F.when(F.col("has_title") & F.col("has_message"), "title_and_message")
            .when(F.col("has_title"), "title")
            .otherwise("message"),
        )
        .withColumn("is_multi_order_review", F.col("order_count") > 1)
        .withColumn(
            "purchase_month_index",
            F.when(
                F.col("purchase_year").isNotNull()
                & F.col("purchase_month").isNotNull(),
                F.col("purchase_year") * 12 + F.col("purchase_month"),
            ),
        )
        .withColumn(
            "purchase_month",
            F.when(
                F.col("purchase_year").isNotNull()
                & F.col("purchase_month").isNotNull(),
                F.format_string(
                    "%04d-%02d", F.col("purchase_year"), F.col("purchase_month")
                ),
            ),
        )
        .withColumn("document_hash", F.sha2("document_text", 256))
    )
    return documents.select(
        "document_id",
        "document_hash",
        "document_text",
        "review_id",
        "review_comment_title_original",
        "review_comment_message_original",
        "review_score",
        "review_language",
        "review_creation_date",
        "review_answer_timestamp",
        "has_title",
        "has_message",
        "text_source",
        "order_id",
        "order_count",
        "is_multi_order_review",
        "product_id",
        "product_category",
        "customer_state",
        "purchase_month",
        "purchase_month_index",
        "delivery_delay_days",
        "is_late",
        "item_count",
        "distinct_product_count",
        "is_single_product_order",
        "is_single_category_order",
        "themes",
        "theme_classifier_version",
        "theme_classification_method",
    )


def validate_document_contract(
    reviews: DataFrame, documents: DataFrame, review_themes: DataFrame
) -> dict:
    eligible_count = reviews.where(F.col("text_is_eligible")).count()
    eligible_distinct = (
        reviews.where(F.col("text_is_eligible")).select("review_id").distinct().count()
    )
    document_count = documents.count()
    document_distinct = documents.select("document_id").distinct().count()
    if eligible_count != eligible_distinct:
        raise ValueError("Eligible canonical reviews contain duplicate review_id values")
    if document_count != eligible_distinct or document_distinct != document_count:
        raise ValueError("RAG documents do not reconcile with eligible review_id values")
    invalid_text = documents.where(
        F.col("document_text").isNull() | (F.length(F.trim("document_text")) == 0)
    ).count()
    if invalid_text:
        raise ValueError("RAG documents contain empty text")
    duplicate_themes = (
        review_themes.groupBy("review_id", "theme").count().where("count > 1").count()
    )
    unclassified = (
        documents.select("review_id")
        .join(review_themes.select("review_id").distinct(), "review_id", "left_anti")
        .count()
    )
    if duplicate_themes or unclassified:
        raise ValueError("Theme assignments are duplicated or incomplete")
    sources = {
        row.text_source: row["count"]
        for row in documents.groupBy("text_source").count().collect()
    }
    return {
        "eligible_review_ids": eligible_distinct,
        "document_count": document_count,
        "theme_assignment_count": review_themes.count(),
        "text_source_counts": sources,
    }
