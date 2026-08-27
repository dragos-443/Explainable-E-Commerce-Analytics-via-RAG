"""Reproducible exploratory analysis for the Olist curated datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ecommerce_rag.analytics.metrics import (
    NEGATIVE_SCORE_MAX,
    negative_indicator,
)
from ecommerce_rag.common.config import load_config


MIN_GROUP_REVIEWS = 500

THEME_PATTERNS = {
    "non_delivery": r"\b(n[aã]o recebi|n[aã]o chegou|n[aã]o foi entregue|ainda n[aã]o|aguardando|nunca chegou)\b",
    "delivery_delay": r"\b(atras\w*|demor\w*|prazo|correios|entrega tard\w*)\b",
    "wrong_or_missing_item": r"\b(errad\w*|falt\w*|incomplet\w*|veio outro|produto diferente|item diferente)\b",
    "damaged_or_defective": r"\b(defeit\w*|quebrad\w*|danific\w*|avariad\w*|n[aã]o funciona\w*)\b",
    "quality_or_expectation": r"\b(qualidade|ruim|p[eé]ssim\w*|propaganda|foto|tamanho|material|acabamento|diferente do anunciado)\b",
    "service_or_refund": r"\b(atendimento|contato|resposta|retorno|troca|devolu[cç][aã]o|reembolso|estorno|cancel\w*|dinheiro)\b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="local")
    parser.add_argument("--report-root", default="/workspace/reports/eda/phase2")
    return parser.parse_args()


def build_monthly_review_statistics(reviews: DataFrame) -> DataFrame:
    monthly = (
        reviews.where(
            F.col("purchase_year").isNotNull()
            & F.col("purchase_month").isNotNull()
        )
        .groupBy("purchase_year", "purchase_month")
        .agg(
            F.count("*").alias("review_count"),
            F.sum("review_score").alias("rating_sum"),
            F.sum(negative_indicator()).alias("negative_review_count"),
            F.sum(F.col("text_is_eligible").cast("int")).alias(
                "text_eligible_count"
            ),
        )
        .withColumn(
            "month", F.make_date("purchase_year", "purchase_month", F.lit(1))
        )
        .withColumn("average_rating", F.col("rating_sum") / F.col("review_count"))
        .withColumn(
            "negative_review_rate",
            F.col("negative_review_count") / F.col("review_count"),
        )
        .withColumn(
            "text_coverage", F.col("text_eligible_count") / F.col("review_count")
        )
        .withColumn(
            "eligible_for_temporal_comparison",
            F.col("review_count") >= MIN_GROUP_REVIEWS,
        )
    )
    eligible = monthly.where("eligible_for_temporal_comparison")
    baseline_window = Window.orderBy("month").rowsBetween(-3, -1)
    eligible = (
        eligible.withColumn(
            "baseline_month_count", F.count("*").over(baseline_window)
        )
        .withColumn(
            "baseline_rating_sum", F.sum("rating_sum").over(baseline_window)
        )
        .withColumn(
            "baseline_review_count", F.sum("review_count").over(baseline_window)
        )
        .withColumn(
            "baseline_average_rating",
            F.when(
                F.col("baseline_month_count") == 3,
                F.col("baseline_rating_sum") / F.col("baseline_review_count"),
            ),
        )
        .withColumn(
            "rating_variation",
            F.col("average_rating") - F.col("baseline_average_rating"),
        )
        .drop("baseline_rating_sum", "baseline_review_count")
    )
    return monthly.where("not eligible_for_temporal_comparison").unionByName(
        eligible, allowMissingColumns=True
    ).orderBy("month")


def classify_complaint_themes(reviews: DataFrame) -> DataFrame:
    normalized = F.lower(
        F.regexp_replace(F.coalesce(F.col("review_text"), F.lit("")), r"[^\p{L}]+", " ")
    )
    token_count = F.size(F.filter(F.split(F.trim(normalized), r"\s+"), lambda x: F.length(x) > 1))
    theme_columns = [
        F.when(normalized.rlike(pattern), F.lit(theme))
        for theme, pattern in THEME_PATTERNS.items()
    ]
    matched = F.filter(F.array(*theme_columns), lambda x: x.isNotNull())
    return (
        reviews.where(
            (F.col("review_score") <= NEGATIVE_SCORE_MAX)
            & F.col("text_is_eligible")
        )
        .withColumn("normalized_review_text", F.trim(normalized))
        .withColumn("matched_themes", matched)
        .withColumn(
            "complaint_themes",
            F.when(F.size("matched_themes") > 0, F.col("matched_themes"))
            .when(token_count < 4, F.array(F.lit("uncertain")))
            .otherwise(F.array(F.lit("other"))),
        )
        .drop("matched_themes")
    )


def build_tables(orders: DataFrame, reviews: DataFrame, links: DataFrame) -> dict[str, DataFrame]:
    rating_distribution = (
        reviews.groupBy("review_score")
        .agg(F.count("*").alias("review_count"))
        .withColumn(
            "share", F.col("review_count") / F.sum("review_count").over(Window.partitionBy())
        )
        .orderBy("review_score")
    )

    category_statistics = (
        reviews.where(F.col("product_category").isNotNull())
        .groupBy("product_category")
        .agg(
            F.count("*").alias("review_count"),
            F.avg("review_score").alias("average_rating"),
            F.avg(negative_indicator()).alias("negative_review_rate"),
            F.avg(F.col("text_is_eligible").cast("int")).alias("text_coverage"),
        )
        .withColumn(
            "eligible_for_comparison", F.col("review_count") >= MIN_GROUP_REVIEWS
        )
    )

    monthly_reviews = build_monthly_review_statistics(reviews)
    monthly_orders = (
        orders.where(F.col("purchase_date").isNotNull())
        .groupBy(F.trunc("purchase_date", "month").alias("month"))
        .agg(
            F.countDistinct("order_id").alias("order_volume"),
            F.avg("order_value").alias("average_order_value"),
            F.avg("delivery_time_days").alias("average_delivery_time_days"),
            F.avg(F.col("is_late").cast("int")).alias("late_delivery_rate"),
        )
        .orderBy("month")
    )

    single_order_reviews = reviews.where(F.col("linked_order_count") == 1).select(
        "review_id",
        "review_score",
        F.element_at("linked_order_ids", 1).alias("order_id"),
    )
    review_delivery = single_order_reviews.join(
        orders.select(
            "order_id", "is_late", "delivery_delay_days", "delivery_time_days"
        ),
        "order_id",
        "inner",
    )
    delivery_by_rating = (
        review_delivery.groupBy("review_score")
        .agg(
            F.count("*").alias("linked_review_count"),
            F.count("is_late").alias("delivery_eligible_count"),
            F.avg(F.col("is_late").cast("int")).alias("late_delivery_rate"),
            F.avg("delivery_time_days").alias("average_delivery_time_days"),
            F.avg(
                F.when(F.col("is_late"), F.col("delivery_delay_days"))
            ).alias("average_delivery_delay"),
        )
        .orderBy("review_score")
    )

    status_distribution = (
        orders.groupBy("order_status")
        .agg(F.count("*").alias("order_count"))
        .withColumn(
            "share", F.col("order_count") / F.sum("order_count").over(Window.partitionBy())
        )
        .orderBy(F.desc("order_count"))
    )
    geography_statistics = (
        reviews.where(F.col("customer_state").isNotNull())
        .groupBy("customer_state")
        .agg(
            F.count("*").alias("review_count"),
            F.avg("review_score").alias("average_rating"),
            F.avg(negative_indicator()).alias("negative_review_rate"),
        )
        .withColumn(
            "eligible_for_comparison", F.col("review_count") >= MIN_GROUP_REVIEWS
        )
    )
    text_coverage_by_rating = reviews.groupBy("review_score").agg(
        F.count("*").alias("review_count"),
        F.sum(F.col("text_is_eligible").cast("int")).alias("text_eligible_count"),
        F.avg(F.col("text_is_eligible").cast("int")).alias("text_coverage"),
    ).orderBy("review_score")

    classified = classify_complaint_themes(reviews).persist(StorageLevel.MEMORY_AND_DISK)
    theme_assignments = classified.select(
        "review_id", "review_score", F.explode("complaint_themes").alias("complaint_theme")
    )
    negative_text_count = classified.count()
    theme_prevalence = (
        theme_assignments.groupBy("complaint_theme")
        .agg(F.countDistinct("review_id").alias("review_count"))
        .withColumn("denominator_negative_text_reviews", F.lit(negative_text_count))
        .withColumn("prevalence", F.col("review_count") / F.lit(negative_text_count))
        .orderBy(F.desc("prevalence"), "complaint_theme")
    )

    category_theme_denominator = (
        classified.where(F.col("product_category").isNotNull())
        .groupBy("product_category")
        .agg(F.countDistinct("review_id").alias("negative_text_review_count"))
    )
    complaint_themes_by_category = (
        classified.where(F.col("product_category").isNotNull())
        .select(
            "review_id",
            "product_category",
            F.explode("complaint_themes").alias("complaint_theme"),
        )
        .groupBy("product_category", "complaint_theme")
        .agg(F.countDistinct("review_id").alias("review_count"))
        .join(category_theme_denominator, "product_category")
        .withColumn(
            "prevalence",
            F.col("review_count") / F.col("negative_text_review_count"),
        )
    )

    month_theme_denominator = (
        classified.where(
            F.col("purchase_year").isNotNull()
            & F.col("purchase_month").isNotNull()
        )
        .groupBy("purchase_year", "purchase_month")
        .agg(F.countDistinct("review_id").alias("negative_text_review_count"))
    )
    complaint_themes_by_month = (
        classified.where(
            F.col("purchase_year").isNotNull()
            & F.col("purchase_month").isNotNull()
        )
        .select(
            "review_id",
            "purchase_year",
            "purchase_month",
            F.explode("complaint_themes").alias("complaint_theme"),
        )
        .groupBy("purchase_year", "purchase_month", "complaint_theme")
        .agg(F.countDistinct("review_id").alias("review_count"))
        .join(month_theme_denominator, ["purchase_year", "purchase_month"])
        .withColumn(
            "prevalence",
            F.col("review_count") / F.col("negative_text_review_count"),
        )
    )

    return {
        "rating_distribution": rating_distribution,
        "category_statistics": category_statistics,
        "monthly_review_statistics": monthly_reviews,
        "monthly_order_statistics": monthly_orders,
        "delivery_by_rating": delivery_by_rating,
        "order_status_distribution": status_distribution,
        "geography_statistics": geography_statistics,
        "text_coverage_by_rating": text_coverage_by_rating,
        "complaint_theme_assignments": theme_assignments,
        "complaint_theme_prevalence": theme_prevalence,
        "complaint_themes_by_category": complaint_themes_by_category,
        "complaint_themes_by_month": complaint_themes_by_month,
    }


def scalar_summary(orders: DataFrame, reviews: DataFrame) -> dict[str, Any]:
    rating = reviews.agg(
        F.count("*").alias("review_count"),
        F.avg("review_score").alias("average_rating"),
        F.sum(negative_indicator()).alias("negative_review_count"),
        F.avg(negative_indicator()).alias("negative_review_rate"),
        F.sum(F.col("text_is_eligible").cast("int")).alias("text_eligible_count"),
        F.avg(F.col("text_is_eligible").cast("int")).alias("text_coverage"),
    ).first().asDict()
    delivery = orders.agg(
        F.count("*").alias("order_count"),
        F.count("is_late").alias("delivery_eligible_count"),
        F.avg(F.col("is_late").cast("int")).alias("late_delivery_rate"),
        F.avg(F.when(F.col("is_late"), F.col("delivery_delay_days"))).alias(
            "average_delivery_delay"
        ),
        F.avg("order_value").alias("average_order_value"),
    ).first().asDict()
    return {**rating, **delivery}


def json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "as_tuple"):
        return float(value)
    return value


def write_local_table(frame: DataFrame, path: Path) -> None:
    rows = frame.collect()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=frame.columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_value(value) for key, value in row.asDict().items()})


def save_figures(tables: dict[str, DataFrame], figure_root: Path) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    rating = tables["rating_distribution"].collect()
    plt.figure(figsize=(7, 4))
    plt.bar([str(x.review_score) for x in rating], [x.review_count for x in rating])
    plt.xlabel("Punteggio")
    plt.ylabel("Recensioni")
    plt.title("Distribuzione dei rating")
    plt.tight_layout()
    plt.savefig(figure_root / "rating_distribution.png", dpi=160)
    plt.close()

    monthly = tables["monthly_review_statistics"].where(
        "eligible_for_temporal_comparison"
    ).orderBy("month").collect()
    months = [x.month for x in monthly]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(months, [x.average_rating for x in monthly], marker="o", label="Rating medio")
    axis.set_ylabel("Rating medio")
    axis.set_ylim(3.5, 4.5)
    second = axis.twinx()
    second.plot(months, [100 * x.negative_review_rate for x in monthly], color="tab:red", marker="s", label="Recensioni negative")
    second.set_ylabel("Recensioni negative (%)")
    axis.set_title("Rating e recensioni negative nel tempo")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_root / "monthly_rating_trend.png", dpi=160)
    plt.close(fig)

    delivery = tables["delivery_by_rating"].collect()
    plt.figure(figsize=(7, 4))
    plt.bar([str(x.review_score) for x in delivery], [100 * x.late_delivery_rate for x in delivery])
    plt.xlabel("Punteggio recensione")
    plt.ylabel("Consegne in ritardo (%)")
    plt.title("Associazione tra rating e consegna tardiva")
    plt.tight_layout()
    plt.savefig(figure_root / "late_delivery_by_rating.png", dpi=160)
    plt.close()

    categories = tables["category_statistics"].where("eligible_for_comparison").orderBy(F.desc("negative_review_rate")).limit(15).collect()
    categories = list(reversed(categories))
    plt.figure(figsize=(9, 6))
    plt.barh([x.product_category for x in categories], [100 * x.negative_review_rate for x in categories])
    plt.xlabel("Recensioni negative (%)")
    plt.title("Categorie con maggiore quota negativa (almeno 500 recensioni)")
    plt.tight_layout()
    plt.savefig(figure_root / "negative_rate_by_category.png", dpi=160)
    plt.close()

    themes = tables["complaint_theme_prevalence"].collect()
    themes = list(reversed(themes))
    plt.figure(figsize=(8, 5))
    plt.barh([x.complaint_theme for x in themes], [100 * x.prevalence for x in themes])
    plt.xlabel("Prevalenza sulle recensioni negative con testo (%)")
    plt.title("Temi preliminari dei reclami (multi-label)")
    plt.tight_layout()
    plt.savefig(figure_root / "complaint_theme_prevalence.png", dpi=160)
    plt.close()

    orders = tables["monthly_order_statistics"].orderBy("month").collect()
    plt.figure(figsize=(10, 4))
    plt.plot([x.month for x in orders], [x.order_volume for x in orders], marker="o")
    plt.ylabel("Ordini")
    plt.title("Volume degli ordini nel tempo")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(figure_root / "monthly_order_volume.png", dpi=160)
    plt.close()


def derive_findings(tables: dict[str, DataFrame], summary: dict[str, Any]) -> list[dict[str, Any]]:
    delivery = {row.review_score: row for row in tables["delivery_by_rating"].collect()}
    worst_month = (
        tables["monthly_review_statistics"]
        .where("rating_variation is not null")
        .orderBy(F.desc("negative_review_rate"))
        .first()
    )
    worst_category = (
        tables["category_statistics"]
        .where("eligible_for_comparison")
        .orderBy(F.desc("negative_review_rate"))
        .first()
    )
    return [
        {
            "id": "rating_delivery_association",
            "phenomenon": "Le recensioni basse sono fortemente associate a consegne tardive.",
            "evidence": {
                "late_rate_rating_1": delivery[1].late_delivery_rate,
                "late_rate_rating_5": delivery[5].late_delivery_rate,
                "difference_percentage_points": 100 * (delivery[1].late_delivery_rate - delivery[5].late_delivery_rate),
            },
            "interpretation_limit": "Associazione osservata; non dimostra causalita per ogni recensione.",
        },
        {
            "id": "worst_month_against_baseline",
            "phenomenon": "Un mese completo mostra un deterioramento marcato rispetto alla baseline mobile.",
            "evidence": {key: json_value(value) for key, value in worst_month.asDict().items()},
            "interpretation_limit": "La baseline descrive i tre mesi precedenti e non controlla stagionalita o composizione.",
        },
        {
            "id": "high_negative_category",
            "phenomenon": "Una categoria con volume sufficiente presenta una quota negativa nettamente superiore al dato complessivo.",
            "evidence": {
                **{key: json_value(value) for key, value in worst_category.asDict().items()},
                "overall_negative_review_rate": summary["negative_review_rate"],
            },
            "interpretation_limit": "Il confronto e descrittivo e usa solo ordini con categoria non ambigua.",
        },
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.environment)
    spark = SparkSession.builder.appName("phase-2-olist-eda").config(
        "spark.sql.shuffle.partitions", "8"
    ).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        curated = f"{config['storage']['curated_uri']}/olist"
        orders = spark.read.parquet(f"{curated}/orders_enriched").cache()
        reviews = spark.read.parquet(f"{curated}/reviews_enriched").cache()
        links = spark.read.parquet(f"{curated}/review_order_links")
        orders.count()
        reviews.count()

        tables = build_tables(orders, reviews, links)
        output_uri = f"{config['storage']['outputs_uri']}/eda/phase2"
        table_root = Path(args.report_root) / "tables"
        for name, frame in tables.items():
            frame.persist(StorageLevel.MEMORY_AND_DISK)
            frame.count()
            frame.write.mode("overwrite").parquet(f"{output_uri}/{name}")
            write_local_table(frame, table_root / f"{name}.csv")

        summary = scalar_summary(orders, reviews)
        findings = derive_findings(tables, summary)
        report_root = Path(args.report_root)
        report_root.mkdir(parents=True, exist_ok=True)
        with (report_root / "kpi_summary.json").open("w", encoding="utf-8") as output:
            json.dump({key: json_value(value) for key, value in summary.items()}, output, ensure_ascii=False, indent=2)
            output.write("\n")
        with (report_root / "candidate_findings.json").open("w", encoding="utf-8") as output:
            json.dump(findings, output, ensure_ascii=False, indent=2)
            output.write("\n")
        save_figures(tables, report_root / "figures")
        print("PHASE2_SUMMARY=" + json.dumps({"tables": sorted(tables), "findings": findings}, ensure_ascii=False, sort_keys=True))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
