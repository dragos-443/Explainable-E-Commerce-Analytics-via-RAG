"""Olist ingestion and preprocessing pipeline for Phase 1."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Iterable

from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DecimalType

from ecommerce_rag.common.config import load_config
from ecommerce_rag.preprocessing.schemas import RAW_FILES, raw_schema


MONEY = DecimalType(18, 2)
TIMESTAMP_PATTERN = "yyyy-MM-dd HH:mm:ss"


@dataclass
class DatasetContract:
    layer: str
    name: str
    grain: str
    logical_key: list[str]
    required_columns: list[str]
    nullable_columns: list[str]
    deduplication: str
    invariants: list[str]
    derivations: str


class QualityRecorder:
    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []
        self.null_profile: list[dict[str, Any]] = []

    def add(
        self,
        dataset: str,
        metric: str,
        value: int,
        status: str = "observed",
        details: str = "",
    ) -> None:
        self.metrics.append(
            {
                "dataset": dataset,
                "metric": metric,
                "value": int(value),
                "status": status,
                "details": details,
            }
        )

    def profile_nulls(self, dataset: str, frame: DataFrame, total: int) -> None:
        expressions = [
            F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
            for column in frame.columns
        ]
        counts = frame.agg(*expressions).first().asDict()
        for column, null_count in counts.items():
            self.null_profile.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "null_count": int(null_count),
                    "total_count": int(total),
                    "null_rate": float(null_count / total) if total else 0.0,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="local")
    return parser.parse_args()


def trimmed(column: str) -> Column:
    value = F.trim(F.col(column))
    return F.when(value == "", None).otherwise(value)


def normalized_text_category(column: str) -> Column:
    return F.lower(trimmed(column))


def normalized_state(column: str) -> Column:
    return F.upper(trimmed(column))


def read_raw(spark: SparkSession, raw_uri: str, dataset: str) -> DataFrame:
    reader = (
        spark.read.option("header", True)
        .option("mode", "FAILFAST")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .schema(raw_schema(dataset))
    )
    if dataset == "reviews":
        reader = reader.option("multiLine", True)
    return reader.csv(f"{raw_uri}/olist/{RAW_FILES[dataset]}")


def normalize(raw: dict[str, DataFrame]) -> dict[str, DataFrame]:
    translation = raw["category_translation"].select(
        normalized_text_category("product_category_name").alias(
            "product_category_name"
        ),
        normalized_text_category("product_category_name_english").alias(
            "product_category_name_english"
        ),
    )

    products_base = raw["products"].select(
        trimmed("product_id").alias("product_id"),
        normalized_text_category("product_category_name").alias(
            "product_category_name_original"
        ),
        trimmed("product_name_lenght").cast("int").alias("product_name_length"),
        trimmed("product_description_lenght")
        .cast("int")
        .alias("product_description_length"),
        trimmed("product_photos_qty").cast("int").alias("product_photos_qty"),
        trimmed("product_weight_g").cast("int").alias("product_weight_g"),
        trimmed("product_length_cm").cast("int").alias("product_length_cm"),
        trimmed("product_height_cm").cast("int").alias("product_height_cm"),
        trimmed("product_width_cm").cast("int").alias("product_width_cm"),
    )
    products = (
        products_base.join(
            translation,
            products_base.product_category_name_original
            == translation.product_category_name,
            "left",
        )
        .drop(translation.product_category_name)
        .withColumn(
            "product_category",
            F.coalesce(
                "product_category_name_english", "product_category_name_original"
            ),
        )
        .withColumn(
            "product_category_source",
            F.when(F.col("product_category_name_english").isNotNull(), F.lit("english"))
            .when(
                F.col("product_category_name_original").isNotNull(),
                F.lit("portuguese_fallback"),
            )
            .otherwise(F.lit(None).cast("string")),
        )
    )

    return {
        "category_translation": translation,
        "customers": raw["customers"].select(
            trimmed("customer_id").alias("customer_id"),
            trimmed("customer_unique_id").alias("customer_unique_id"),
            trimmed("customer_zip_code_prefix")
            .cast("int")
            .alias("customer_zip_code_prefix"),
            normalized_text_category("customer_city").alias("customer_city"),
            normalized_state("customer_state").alias("customer_state"),
        ),
        "geolocation": raw["geolocation"].select(
            trimmed("geolocation_zip_code_prefix")
            .cast("int")
            .alias("geolocation_zip_code_prefix"),
            trimmed("geolocation_lat").cast("double").alias("geolocation_lat"),
            trimmed("geolocation_lng").cast("double").alias("geolocation_lng"),
            normalized_text_category("geolocation_city").alias("geolocation_city"),
            normalized_state("geolocation_state").alias("geolocation_state"),
        ),
        "order_items": raw["order_items"].select(
            trimmed("order_id").alias("order_id"),
            trimmed("order_item_id").cast("int").alias("order_item_id"),
            trimmed("product_id").alias("product_id"),
            trimmed("seller_id").alias("seller_id"),
            F.to_timestamp(trimmed("shipping_limit_date"), TIMESTAMP_PATTERN).alias(
                "shipping_limit_date"
            ),
            trimmed("price").cast(MONEY).alias("price"),
            trimmed("freight_value").cast(MONEY).alias("freight_value"),
        ),
        "payments": raw["payments"].select(
            trimmed("order_id").alias("order_id"),
            trimmed("payment_sequential").cast("int").alias("payment_sequential"),
            normalized_text_category("payment_type").alias("payment_type"),
            trimmed("payment_installments")
            .cast("int")
            .alias("payment_installments"),
            trimmed("payment_value").cast(MONEY).alias("payment_value"),
        ),
        "reviews": raw["reviews"].select(
            trimmed("review_id").alias("review_id"),
            trimmed("order_id").alias("order_id"),
            trimmed("review_score").cast("int").alias("review_score"),
            F.col("review_comment_title").alias("review_comment_title_original"),
            F.col("review_comment_message").alias("review_comment_message_original"),
            F.to_timestamp(trimmed("review_creation_date"), TIMESTAMP_PATTERN).alias(
                "review_creation_date"
            ),
            F.to_timestamp(trimmed("review_answer_timestamp"), TIMESTAMP_PATTERN).alias(
                "review_answer_timestamp"
            ),
        ),
        "orders": raw["orders"].select(
            trimmed("order_id").alias("order_id"),
            trimmed("customer_id").alias("customer_id"),
            normalized_text_category("order_status").alias("order_status"),
            F.to_timestamp(trimmed("order_purchase_timestamp"), TIMESTAMP_PATTERN).alias(
                "order_purchase_timestamp"
            ),
            F.to_timestamp(trimmed("order_approved_at"), TIMESTAMP_PATTERN).alias(
                "order_approved_at"
            ),
            F.to_timestamp(
                trimmed("order_delivered_carrier_date"), TIMESTAMP_PATTERN
            ).alias("order_delivered_carrier_date"),
            F.to_timestamp(
                trimmed("order_delivered_customer_date"), TIMESTAMP_PATTERN
            ).alias("order_delivered_customer_date"),
            F.to_timestamp(
                trimmed("order_estimated_delivery_date"), TIMESTAMP_PATTERN
            ).alias("order_estimated_delivery_date"),
        ),
        "products": products,
        "sellers": raw["sellers"].select(
            trimmed("seller_id").alias("seller_id"),
            trimmed("seller_zip_code_prefix")
            .cast("int")
            .alias("seller_zip_code_prefix"),
            normalized_text_category("seller_city").alias("seller_city"),
            normalized_state("seller_state").alias("seller_state"),
        ),
    }


def assert_unique(
    frame: DataFrame,
    dataset: str,
    key: list[str],
    recorder: QualityRecorder,
) -> None:
    null_key_condition = F.col(key[0]).isNull()
    for column in key[1:]:
        null_key_condition = null_key_condition | F.col(column).isNull()
    null_keys = frame.where(null_key_condition).count()
    recorder.add(
        dataset,
        "null_logical_key_rows",
        null_keys,
        "pass" if null_keys == 0 else "fail",
        ",".join(key),
    )
    if null_keys:
        raise ValueError(f"{dataset} contains {null_keys} rows with null keys: {key}")

    duplicates = (
        frame.groupBy(*key).count().where(F.col("count") > 1).count()
    )
    recorder.add(
        dataset,
        "duplicate_logical_keys",
        duplicates,
        "pass" if duplicates == 0 else "fail",
        ",".join(key),
    )
    if duplicates:
        raise ValueError(f"{dataset} contains {duplicates} duplicated keys: {key}")


def validate_foreign_key(
    child: DataFrame,
    parent: DataFrame,
    child_column: str,
    parent_column: str,
    relationship: str,
    recorder: QualityRecorder,
) -> None:
    orphans = (
        child.select(F.col(child_column).alias("fk"))
        .where(F.col("fk").isNotNull())
        .distinct()
        .join(
            parent.select(F.col(parent_column).alias("fk")).distinct(),
            "fk",
            "left_anti",
        )
        .count()
    )
    recorder.add(
        relationship,
        "orphan_distinct_keys",
        orphans,
        "pass" if orphans == 0 else "fail",
    )
    if orphans:
        raise ValueError(f"Foreign-key validation failed for {relationship}: {orphans}")


def join_without_fanout(
    left: DataFrame,
    right: DataFrame,
    on: str,
    label: str,
    recorder: QualityRecorder,
) -> DataFrame:
    before = left.count()
    joined = left.join(right, on, "left")
    after = joined.count()
    recorder.add(label, "rows_before_join", before)
    recorder.add(
        label,
        "rows_after_join",
        after,
        "pass" if after == before else "fail",
    )
    if after != before:
        raise ValueError(f"Fan-out detected in {label}: before={before}, after={after}")
    return joined


def language_expression(text_column: Column) -> Column:
    clean = F.trim(F.regexp_replace(F.lower(F.coalesce(text_column, F.lit(""))), r"[^\p{L}]+", " "))
    tokens = F.array_distinct(F.split(clean, r"\s+"))

    def score(words: Iterable[str]) -> Column:
        return F.size(F.array_intersect(tokens, F.array(*[F.lit(word) for word in words])))

    pt = score(
        [
            "não", "nao", "muito", "produto", "entrega", "chegou", "com",
            "para", "que", "uma", "bom", "ruim", "recebi", "prazo", "ainda",
            "foi", "mas", "meu", "minha", "comprar", "recomendo",
        ]
    )
    es = score(
        [
            "no", "muy", "producto", "entrega", "llegó", "con", "para", "que",
            "una", "bueno", "malo", "recibí", "plazo", "todavía", "pero",
            "recomiendo", "comprar",
        ]
    )
    en = score(
        [
            "not", "very", "product", "delivery", "arrived", "with", "for", "the",
            "good", "bad", "received", "late", "still", "but", "recommend", "buy",
        ]
    )
    it = score(
        [
            "non", "molto", "prodotto", "consegna", "arrivato", "con", "per",
            "che", "una", "buono", "cattivo", "ricevuto", "ritardo", "ancora",
            "ma", "consiglio", "comprare",
        ]
    )
    max_score = F.greatest(pt, es, en, it)
    token_count = F.size(F.filter(tokens, lambda token: F.length(token) > 1))
    unique_winner = (
        (F.when(pt == max_score, 1).otherwise(0))
        + (F.when(es == max_score, 1).otherwise(0))
        + (F.when(en == max_score, 1).otherwise(0))
        + (F.when(it == max_score, 1).otherwise(0))
    ) == 1

    return (
        F.when((token_count < 3) | (max_score < 2) | (~unique_winner), "unknown_or_ambiguous")
        .when(pt == max_score, "pt")
        .when(es == max_score, "es")
        .when(en == max_score, "en")
        .when(it == max_score, "it")
        .otherwise("unknown_or_ambiguous")
    )


def build_intermediates(
    processed: dict[str, DataFrame], recorder: QualityRecorder
) -> dict[str, DataFrame]:
    items = processed["order_items"]
    products = processed["products"].select("product_id", "product_category")
    sellers = processed["sellers"].select("seller_id", "seller_state")
    item_count = items.count()
    item_details = items.join(products, "product_id", "left")
    if item_details.count() != item_count:
        raise ValueError("Fan-out joining order_items to products")
    item_details = item_details.join(sellers, "seller_id", "left")
    if item_details.count() != item_count:
        raise ValueError("Fan-out joining order_items to sellers")
    recorder.add("item_details", "rows_after_dimension_joins", item_count, "pass")

    items_by_order = (
        item_details.groupBy("order_id")
        .agg(
            F.count("*").alias("item_count"),
            F.countDistinct("product_id").alias("distinct_product_count"),
            F.countDistinct("seller_id").alias("distinct_seller_count"),
            F.countDistinct("product_category").alias("distinct_known_category_count"),
            F.sum(F.when(F.col("product_category").isNull(), 1).otherwise(0))
            .cast("int")
            .alias("unknown_category_item_count"),
            F.sum("price").cast(MONEY).alias("item_subtotal"),
            F.sum("freight_value").cast(MONEY).alias("freight_total"),
            F.sort_array(F.collect_set("product_id")).alias("product_ids"),
            F.sort_array(F.collect_set("seller_id")).alias("seller_ids"),
            F.sort_array(F.collect_set("product_category")).alias("product_categories"),
            F.sort_array(F.collect_set("seller_state")).alias("seller_states"),
        )
        .withColumn(
            "order_value",
            (F.col("item_subtotal") + F.col("freight_total")).cast(MONEY),
        )
        .withColumn(
            "is_strict_mono_category",
            (F.col("unknown_category_item_count") == 0)
            & (F.col("distinct_known_category_count") == 1),
        )
        .withColumn(
            "order_category",
            F.when(
                F.col("is_strict_mono_category"),
                F.element_at("product_categories", 1),
            ),
        )
    )

    payments_by_order = processed["payments"].groupBy("order_id").agg(
        F.count("*").alias("payment_sequence_count"),
        F.countDistinct("payment_type").alias("distinct_payment_type_count"),
        F.sort_array(F.collect_set("payment_type")).alias("payment_types"),
        F.max("payment_installments").alias("max_payment_installments"),
        F.sum("payment_value").cast(MONEY).alias("payment_total"),
    )

    geo = processed["geolocation"].where(
        F.col("geolocation_zip_code_prefix").isNotNull()
    )
    place_counts = geo.groupBy(
        "geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"
    ).agg(F.count("*").alias("place_observation_count"))
    place_window = Window.partitionBy("geolocation_zip_code_prefix").orderBy(
        F.desc("place_observation_count"),
        F.asc_nulls_last("geolocation_city"),
        F.asc_nulls_last("geolocation_state"),
    )
    representative_place = (
        place_counts.withColumn("row_number", F.row_number().over(place_window))
        .where(F.col("row_number") == 1)
        .drop("row_number")
    )
    coordinates = geo.groupBy("geolocation_zip_code_prefix").agg(
        F.avg("geolocation_lat").alias("geolocation_lat"),
        F.avg("geolocation_lng").alias("geolocation_lng"),
        F.count("*").alias("geolocation_observation_count"),
    )
    geolocation_by_zip = coordinates.join(
        representative_place, "geolocation_zip_code_prefix", "left"
    )

    reviews = processed["reviews"]
    payload = F.to_json(
        F.struct(
            "review_score",
            "review_comment_title_original",
            "review_comment_message_original",
            "review_creation_date",
            "review_answer_timestamp",
        ),
        options={"ignoreNullFields": "false"},
    )
    inconsistent_reviews = (
        reviews.withColumn("payload", payload)
        .groupBy("review_id")
        .agg(F.countDistinct("payload").alias("payload_count"))
        .where(F.col("payload_count") > 1)
        .count()
    )
    recorder.add(
        "reviews",
        "review_ids_with_inconsistent_payload",
        inconsistent_reviews,
        "pass" if inconsistent_reviews == 0 else "fail",
    )
    if inconsistent_reviews:
        raise ValueError("Repeated review_id values have inconsistent payloads")

    review_window = Window.partitionBy("review_id").orderBy("order_id")
    reviews_canonical = (
        reviews.withColumn("row_number", F.row_number().over(review_window))
        .where(F.col("row_number") == 1)
        .drop("row_number", "order_id")
    )
    title_present = F.length(F.trim(F.coalesce(F.col("review_comment_title_original"), F.lit("")))) > 0
    message_present = F.length(F.trim(F.coalesce(F.col("review_comment_message_original"), F.lit("")))) > 0
    reviews_canonical = (
        reviews_canonical.withColumn(
            "review_text",
            F.when(
                title_present | message_present,
                F.concat_ws(
                    "\n\n",
                    F.when(title_present, F.col("review_comment_title_original")),
                    F.when(message_present, F.col("review_comment_message_original")),
                ),
            ),
        )
        .withColumn("has_title", title_present)
        .withColumn("has_message", message_present)
        .withColumn("text_is_eligible", title_present | message_present)
        .withColumn("review_language", language_expression(F.col("review_text")))
        .withColumn("language_detection_method", F.lit("heuristic_stopwords_v1"))
    )
    review_order_links = reviews.select("review_id", "order_id").distinct()

    return {
        "items_by_order": items_by_order,
        "payments_by_order": payments_by_order,
        "geolocation_by_zip": geolocation_by_zip,
        "reviews_canonical": reviews_canonical,
        "review_order_links": review_order_links,
    }


def build_curated(
    processed: dict[str, DataFrame],
    intermediate: dict[str, DataFrame],
    recorder: QualityRecorder,
) -> dict[str, DataFrame]:
    orders = processed["orders"]
    orders_enriched = join_without_fanout(
        orders,
        intermediate["items_by_order"],
        "order_id",
        "orders_join_items",
        recorder,
    )
    orders_enriched = join_without_fanout(
        orders_enriched,
        intermediate["payments_by_order"],
        "order_id",
        "orders_join_payments",
        recorder,
    )
    orders_enriched = join_without_fanout(
        orders_enriched,
        processed["customers"],
        "customer_id",
        "orders_join_customers",
        recorder,
    )
    orders_enriched = orders_enriched.join(
        intermediate["geolocation_by_zip"],
        orders_enriched.customer_zip_code_prefix
        == intermediate["geolocation_by_zip"].geolocation_zip_code_prefix,
        "left",
    ).drop(intermediate["geolocation_by_zip"].geolocation_zip_code_prefix)
    final_order_count = orders_enriched.count()
    original_order_count = orders.count()
    recorder.add(
        "orders_join_geolocation",
        "rows_after_join",
        final_order_count,
        "pass" if final_order_count == original_order_count else "fail",
    )
    if final_order_count != original_order_count:
        raise ValueError("Fan-out detected joining orders to geolocation_by_zip")

    seconds_per_day = F.lit(86400.0)
    orders_enriched = (
        orders_enriched.withColumn(
            "delivery_time_days",
            F.round(
                (
                    F.col("order_delivered_customer_date").cast("long")
                    - F.col("order_purchase_timestamp").cast("long")
                )
                / seconds_per_day,
                3,
            ),
        )
        .withColumn(
            "delivery_delay_days",
            F.round(
                (
                    F.col("order_delivered_customer_date").cast("long")
                    - F.col("order_estimated_delivery_date").cast("long")
                )
                / seconds_per_day,
                3,
            ),
        )
        .withColumn(
            "is_late",
            F.when(
                F.col("order_delivered_customer_date").isNotNull()
                & F.col("order_estimated_delivery_date").isNotNull(),
                F.col("order_delivered_customer_date")
                > F.col("order_estimated_delivery_date"),
            ),
        )
        .withColumn("purchase_year", F.year("order_purchase_timestamp"))
        .withColumn("purchase_month", F.month("order_purchase_timestamp"))
        .withColumn("purchase_date", F.to_date("order_purchase_timestamp"))
    )

    links_with_order = intermediate["review_order_links"].join(
        orders_enriched.select(
            "order_id",
            "order_category",
            "is_strict_mono_category",
            "customer_state",
            "order_status",
            "purchase_year",
            "purchase_month",
        ),
        "order_id",
        "left",
    )
    if links_with_order.count() != intermediate["review_order_links"].count():
        raise ValueError("Fan-out detected while enriching review-order links")

    review_metadata = links_with_order.groupBy("review_id").agg(
        F.count("*").alias("linked_order_count"),
        F.sort_array(F.collect_set("order_id")).alias("linked_order_ids"),
        F.count("order_category").alias("known_order_category_count"),
        F.countDistinct("order_category").alias("distinct_order_category_count"),
        F.min(F.col("is_strict_mono_category").cast("int")).alias(
            "all_orders_strict_mono_category"
        ),
        F.first("order_category", ignorenulls=True).alias("candidate_category"),
        F.count("customer_state").alias("known_customer_state_count"),
        F.countDistinct("customer_state").alias("distinct_customer_state_count"),
        F.first("customer_state", ignorenulls=True).alias("candidate_customer_state"),
        F.count("order_status").alias("known_order_status_count"),
        F.countDistinct("order_status").alias("distinct_order_status_count"),
        F.first("order_status", ignorenulls=True).alias("candidate_order_status"),
        F.count("purchase_year").alias("known_purchase_year_count"),
        F.countDistinct("purchase_year").alias("distinct_purchase_year_count"),
        F.first("purchase_year", ignorenulls=True).alias("candidate_purchase_year"),
        F.count("purchase_month").alias("known_purchase_month_count"),
        F.countDistinct("purchase_month").alias("distinct_purchase_month_count"),
        F.first("purchase_month", ignorenulls=True).alias("candidate_purchase_month"),
    )
    review_metadata = (
        review_metadata.withColumn(
            "product_category",
            F.when(
                (F.col("known_order_category_count") == F.col("linked_order_count"))
                & (F.col("distinct_order_category_count") == 1)
                & (F.col("all_orders_strict_mono_category") == 1),
                F.col("candidate_category"),
            ),
        )
        .withColumn(
            "customer_state",
            F.when(
                (F.col("known_customer_state_count") == F.col("linked_order_count"))
                & (F.col("distinct_customer_state_count") == 1),
                F.col("candidate_customer_state"),
            ),
        )
        .withColumn(
            "order_status",
            F.when(
                (F.col("known_order_status_count") == F.col("linked_order_count"))
                & (F.col("distinct_order_status_count") == 1),
                F.col("candidate_order_status"),
            ),
        )
        .withColumn(
            "purchase_year",
            F.when(
                (F.col("known_purchase_year_count") == F.col("linked_order_count"))
                & (F.col("distinct_purchase_year_count") == 1),
                F.col("candidate_purchase_year"),
            ),
        )
        .withColumn(
            "purchase_month",
            F.when(
                (F.col("known_purchase_month_count") == F.col("linked_order_count"))
                & (F.col("distinct_purchase_month_count") == 1),
                F.col("candidate_purchase_month"),
            ),
        )
        .drop(
            "candidate_category",
            "candidate_customer_state",
            "candidate_order_status",
            "candidate_purchase_year",
            "candidate_purchase_month",
            "known_order_category_count",
            "distinct_order_category_count",
            "all_orders_strict_mono_category",
            "known_customer_state_count",
            "distinct_customer_state_count",
            "known_order_status_count",
            "distinct_order_status_count",
            "known_purchase_year_count",
            "distinct_purchase_year_count",
            "known_purchase_month_count",
            "distinct_purchase_month_count",
        )
    )

    reviews_enriched = intermediate["reviews_canonical"].join(
        review_metadata, "review_id", "left"
    )
    return {
        "orders_enriched": orders_enriched,
        "reviews_enriched": reviews_enriched,
        "review_order_links": intermediate["review_order_links"],
    }


def contracts() -> dict[tuple[str, str], DatasetContract]:
    base = {
        "orders": ("one row per order", ["order_id"]),
        "order_items": ("one row per item position in an order", ["order_id", "order_item_id"]),
        "payments": ("one row per payment sequence", ["order_id", "payment_sequential"]),
        "reviews": ("one row per review-order association", ["review_id", "order_id"]),
        "products": ("one row per product", ["product_id"]),
        "customers": ("one row per order-level customer id", ["customer_id"]),
        "sellers": ("one row per seller", ["seller_id"]),
        "geolocation": ("one row per observed coordinate for a ZIP prefix", []),
        "category_translation": ("one row per Portuguese category", ["product_category_name"]),
        "items_by_order": ("one row per order with item summaries", ["order_id"]),
        "payments_by_order": ("one row per order with payment summaries", ["order_id"]),
        "geolocation_by_zip": ("one representative row per ZIP prefix", ["geolocation_zip_code_prefix"]),
        "reviews_canonical": ("one row per review id", ["review_id"]),
        "review_order_links": ("one row per unique review-order association", ["review_id", "order_id"]),
    }
    result: dict[tuple[str, str], DatasetContract] = {}
    for name, (grain, key) in base.items():
        result[("processed", name)] = DatasetContract(
            "processed",
            name,
            grain,
            key,
            key,
            [],
            "No implicit deduplication; only declared bridge/canonical rules are applied.",
            ["logical key is unique"] if key else ["raw coordinate observations retained"],
            "Typed casts, deterministic category normalization and documented aggregations.",
        )
    result[("curated", "orders_enriched")] = DatasetContract(
        "curated", "orders_enriched", "one row per order", ["order_id"], ["order_id"],
        ["item summaries", "payment summaries", "delivery features"],
        "Left joins only after dimensions are reduced to one row per join key.",
        ["row count equals orders", "order_id is unique"],
        "delivery times from timestamps; order_value is item_subtotal plus freight_total.",
    )
    result[("curated", "reviews_enriched")] = DatasetContract(
        "curated", "reviews_enriched", "one row per canonical review", ["review_id"], ["review_id"],
        ["text", "non-ambiguous order metadata"],
        "Repeated review ids collapse only after payload consistency is verified.",
        ["review_id is unique", "original title and message are preserved"],
        "Order/category metadata is exposed only when identical across all linked orders.",
    )
    result[("curated", "review_order_links")] = DatasetContract(
        "curated", "review_order_links", "one row per unique review-order association",
        ["review_id", "order_id"], ["review_id", "order_id"], [],
        "Exact duplicate associations are removed.",
        ["composite key is unique"],
        "Bridge between review and order grains; no text duplication.",
    )
    return result


def write_frame(frame: DataFrame, uri: str) -> None:
    frame.write.mode("overwrite").parquet(uri)


def main() -> None:
    args = parse_args()
    config = load_config(args.environment)
    spark = (
        SparkSession.builder.appName("phase-1-olist-preprocessing")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    recorder = QualityRecorder()

    try:
        raw = {
            dataset: read_raw(spark, config["storage"]["raw_uri"], dataset)
            for dataset in RAW_FILES
        }
        processed = normalize(raw)
        primary_keys = {
            "orders": ["order_id"],
            "order_items": ["order_id", "order_item_id"],
            "payments": ["order_id", "payment_sequential"],
            "products": ["product_id"],
            "customers": ["customer_id"],
            "sellers": ["seller_id"],
            "category_translation": ["product_category_name"],
        }

        for name, frame in processed.items():
            frame.persist(StorageLevel.DISK_ONLY)
            total = frame.count()
            recorder.add(name, "row_count", total)
            recorder.profile_nulls(name, frame, total)
            if name in primary_keys:
                assert_unique(frame, name, primary_keys[name], recorder)

        assert_unique(
            processed["reviews"],
            "review_order_associations_raw",
            ["review_id", "order_id"],
            recorder,
        )
        invalid_scores = processed["reviews"].where(
            F.col("review_score").isNull() | (~F.col("review_score").between(1, 5))
        ).count()
        recorder.add(
            "reviews", "invalid_review_scores", invalid_scores,
            "pass" if invalid_scores == 0 else "fail"
        )
        if invalid_scores:
            raise ValueError(f"Invalid review scores: {invalid_scores}")

        validate_foreign_key(processed["orders"], processed["customers"], "customer_id", "customer_id", "orders.customer_id->customers", recorder)
        validate_foreign_key(processed["order_items"], processed["orders"], "order_id", "order_id", "order_items.order_id->orders", recorder)
        validate_foreign_key(processed["order_items"], processed["products"], "product_id", "product_id", "order_items.product_id->products", recorder)
        validate_foreign_key(processed["order_items"], processed["sellers"], "seller_id", "seller_id", "order_items.seller_id->sellers", recorder)
        validate_foreign_key(processed["payments"], processed["orders"], "order_id", "order_id", "payments.order_id->orders", recorder)
        validate_foreign_key(processed["reviews"], processed["orders"], "order_id", "order_id", "reviews.order_id->orders", recorder)

        intermediate = build_intermediates(processed, recorder)
        for name, frame in intermediate.items():
            frame.persist(StorageLevel.DISK_ONLY)
            key = ["review_id", "order_id"] if name == "review_order_links" else [
                "geolocation_zip_code_prefix" if name == "geolocation_by_zip" else
                "review_id" if name == "reviews_canonical" else "order_id"
            ]
            assert_unique(frame, name, key, recorder)

        curated = build_curated(processed, intermediate, recorder)
        assert_unique(curated["orders_enriched"], "orders_enriched", ["order_id"], recorder)
        assert_unique(curated["reviews_enriched"], "reviews_enriched", ["review_id"], recorder)
        assert_unique(curated["review_order_links"], "curated_review_order_links", ["review_id", "order_id"], recorder)

        orders_count = processed["orders"].count()
        orders_enriched_count = curated["orders_enriched"].count()
        if orders_count != orders_enriched_count:
            raise ValueError("orders_enriched does not preserve the orders row count")
        canonical_count = intermediate["reviews_canonical"].count()
        reviews_enriched_count = curated["reviews_enriched"].count()
        if canonical_count != reviews_enriched_count:
            raise ValueError("reviews_enriched does not preserve canonical review count")
        recorder.add("orders_enriched", "row_count", orders_enriched_count, "pass")
        recorder.add("reviews_enriched", "row_count", reviews_enriched_count, "pass")
        recorder.add("review_order_links", "row_count", intermediate["review_order_links"].count(), "pass")

        language_rows = (
            intermediate["reviews_canonical"]
            .groupBy("review_language")
            .count()
            .collect()
        )
        for row in language_rows:
            recorder.add("reviews_canonical", f"language_{row['review_language']}", row["count"])
        recorder.add(
            "reviews_canonical",
            "text_eligible_count",
            intermediate["reviews_canonical"].where("text_is_eligible").count(),
        )

        processed_base_uri = f"{config['storage']['processed_uri']}/olist"
        curated_base_uri = f"{config['storage']['curated_uri']}/olist"
        written: dict[tuple[str, str], DataFrame] = {}
        for name, frame in processed.items():
            write_frame(frame, f"{processed_base_uri}/{name}")
            written[("processed", name)] = frame
        for name, frame in intermediate.items():
            write_frame(frame, f"{processed_base_uri}/{name}")
            written[("processed", name)] = frame
        for name, frame in curated.items():
            write_frame(frame, f"{curated_base_uri}/{name}")
            written[("curated", name)] = frame

        output_uri = f"{config['storage']['outputs_uri']}/quality/phase1"
        spark.createDataFrame(recorder.metrics).coalesce(1).write.mode("overwrite").json(
            f"{output_uri}/metrics"
        )
        spark.createDataFrame(recorder.null_profile).coalesce(1).write.mode("overwrite").parquet(
            f"{output_uri}/null_profile"
        )

        contract_metadata = contracts()
        contract_rows = []
        for key, frame in written.items():
            contract = contract_metadata[key]
            contract_rows.append(
                {
                    "layer": contract.layer,
                    "dataset": contract.name,
                    "grain": contract.grain,
                    "logical_key_json": json.dumps(contract.logical_key),
                    "required_columns_json": json.dumps(contract.required_columns),
                    "nullable_columns_json": json.dumps(contract.nullable_columns),
                    "schema_json": frame.schema.json(),
                    "deduplication": contract.deduplication,
                    "invariants_json": json.dumps(contract.invariants),
                    "derivations": contract.derivations,
                }
            )
        spark.createDataFrame(contract_rows).coalesce(1).write.mode("overwrite").json(
            f"{config['storage']['outputs_uri']}/contracts/phase1"
        )

        summary = {
            "orders": orders_count,
            "orders_enriched": orders_enriched_count,
            "reviews_raw_associations": processed["reviews"].count(),
            "reviews_canonical": canonical_count,
            "reviews_enriched": reviews_enriched_count,
            "review_order_links": intermediate["review_order_links"].count(),
            "items_by_order": intermediate["items_by_order"].count(),
            "payments_by_order": intermediate["payments_by_order"].count(),
            "geolocation_by_zip": intermediate["geolocation_by_zip"].count(),
        }
        print("PHASE1_SUMMARY=" + json.dumps(summary, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
