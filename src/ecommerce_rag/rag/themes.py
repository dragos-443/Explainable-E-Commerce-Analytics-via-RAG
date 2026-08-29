"""Versioned, deterministic complaint-theme classification rules."""

from __future__ import annotations

from typing import Dict

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


THEME_PATTERNS: Dict[str, str] = {
    "non_delivery": r"\b(n[aã]o recebi|n[aã]o chegou|n[aã]o foi entregue|ainda n[aã]o|aguardando|nunca chegou)\b",
    "delivery_delay": (
        r"\b(atras\w*|demor\w*|fora do prazo|ap[oó]s o prazo|depois do prazo|"
        r"prazo (?:j[aá] )?(?:pass\w*|venc\w*|estour\w*)|entrega tard\w*)\b"
    ),
    "wrong_or_missing_item": r"\b(errad\w*|falt\w*|incomplet\w*|veio outro|produto diferente|item diferente)\b",
    "damaged_or_defective": r"\b(defeit\w*|quebrad\w*|danific\w*|avariad\w*|n[aã]o funciona\w*)\b",
    "quality_or_expectation": r"\b(qualidade|ruim|p[eé]ssim\w*|propaganda|foto|tamanho|material|acabamento|diferente do anunciado)\b",
    "service_or_refund": r"\b(atendimento|contato|resposta|retorno|troca|devolu[cç][aã]o|reembolso|estorno|cancel\w*|dinheiro)\b",
}

FALLBACK_THEMES = ("other", "uncertain")
ALL_THEMES = tuple(THEME_PATTERNS) + FALLBACK_THEMES
CLASSIFICATION_METHOD = "deterministic_rule_based_multilabel"


def normalized_review_text(column: str = "review_text") -> Column:
    return F.trim(
        F.lower(
            F.regexp_replace(
                F.coalesce(F.col(column), F.lit("")), r"[^\p{L}]+", " "
            )
        )
    )


def complaint_theme_array(column: str = "review_text") -> Column:
    """Assign one or more themes, always including a deterministic fallback."""
    normalized = normalized_review_text(column)
    token_count = F.size(
        F.filter(
            F.split(normalized, r"\s+"), lambda token: F.length(token) > 1
        )
    )
    candidates = [
        F.when(normalized.rlike(pattern), F.lit(theme))
        for theme, pattern in THEME_PATTERNS.items()
    ]
    matched = F.filter(F.array(*candidates), lambda value: value.isNotNull())
    return (
        F.when(F.size(matched) > 0, matched)
        .when(token_count < 4, F.array(F.lit("uncertain")))
        .otherwise(F.array(F.lit("other")))
    )


def classify_review_themes(
    reviews: DataFrame, classifier_version: str
) -> DataFrame:
    """Classify every eligible textual review and return one row per theme."""
    classified = (
        reviews.where(F.col("text_is_eligible"))
        .select(
            "review_id",
            "review_score",
            complaint_theme_array().alias("themes"),
        )
        .select(
            "review_id",
            "review_score",
            F.explode("themes").alias("theme"),
        )
        .withColumn("classification_score", F.lit(None).cast("double"))
        .withColumn("classifier_version", F.lit(classifier_version))
        .withColumn("classification_method", F.lit(CLASSIFICATION_METHOD))
    )
    return classified.dropDuplicates(["review_id", "theme"])
