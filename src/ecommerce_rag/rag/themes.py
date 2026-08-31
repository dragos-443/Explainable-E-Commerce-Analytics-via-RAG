"""Versioned, deterministic complaint-theme classification rules."""

from __future__ import annotations

from typing import Dict

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


THEME_PATTERNS_V2: Dict[str, str] = {
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

THEME_PATTERNS_V3: Dict[str, str] = {
    "non_delivery": (
        r"\b((?:n[aã]o|n) (?:recebi|recebeu|recebemos) (?:o |a |os |as |meu |minha )?"
        r"(?:produto|pedido|mercadoria|encomenda|compra)|"
        r"(?:produto|pedido|mercadoria|encomenda|compra) n[aã]o (?:chegou|foi entregue)|"
        r"(?:n[aã]o|n) chegou|nunca chegou|"
        r"ainda n[aã]o (?:recebi|chegou|foi entregue)|"
        r"(?:n[aã]o|n) foi entregue)\b"
    ),
    "delivery_delay": (
        r"\b((?:entrega|produto|pedido|mercadoria|encomenda|correios).{0,50}"
        r"(?:atras\w*|demor\w*|fora do prazo|depois do prazo)|"
        r"(?:atras\w*|demor\w*|fora do prazo|depois do prazo).{0,50}"
        r"(?:entrega|produto|pedido|mercadoria|encomenda|correios)|"
        r"prazo (?:j[aá] )?(?:pass\w*|venc\w*|estour\w*)|entrega tard\w*)\b"
    ),
    "wrong_or_missing_item": (
        r"\b((?:recebi|veio|enviaram).{0,50}(?:outro|diferente|marca (?:errada|diferente))|"
        r"(?:produto|item|mercadoria|pedido).{0,30}(?:errad\w*|diferente)|"
        r"(?:falt\w*).{0,30}(?:pe[cç]a|item|produto|unidade|componente|acess[oó]rio)|"
        r"(?:pe[cç]a|item|produto|unidade|componente|acess[oó]rio).{0,30}(?:falt\w*)|"
        r"incomplet\w*)\b"
    ),
    "damaged_or_defective": (
        r"\b(defeit\w*|quebrad\w*|danific\w*|avariad\w*|rachad\w*|"
        r"(?:produto|item|aparelho|rel[oó]gio).{0,30}n[aã]o funciona\w*|"
        r"n[aã]o funciona\w*.{0,30}(?:produto|item|aparelho|rel[oó]gio)|"
        r"parou de funcionar)\b"
    ),
    "quality_or_expectation": (
        r"\b((?:qualidade|material|acabamento).{0,35}"
        r"(?:ruim|p[eé]ssim\w*|baixa|inferior|fr[aá]gil|horr[ií]vel)|"
        r"(?:ruim|p[eé]ssim\w*|baixa|inferior|fr[aá]gil|horr[ií]vel).{0,35}"
        r"(?:qualidade|material|acabamento)|"
        r"diferente (?:da foto|do anunciado|da descri[cç][aã]o)|"
        r"n[aã]o (?:foi|era) (?:a marca|como na foto|como anunciado)|"
        r"falsific\w*)\b"
    ),
    "service_or_refund": (
        r"\b((?:n[aã]o|sem|nunca).{0,35}"
        r"(?:contato|resposta|retorno|atendimento|solu[cç][aã]o|reembolso|estorno|dinheiro)|"
        r"(?:contato|resposta|retorno|atendimento).{0,35}(?:n[aã]o|sem|nunca)|"
        r"aguard\w*.{0,25}(?:retorno|resposta|reembolso|estorno)|"
        r"(?:reembolso|estorno).{0,35}(?:n[aã]o|nunca|aguard\w*|demor\w*)|"
        r"problema.{0,25}n[aã]o resolvid\w*|sem solu[cç][aã]o|"
        r"cancel\w*|devolv\w*)\b"
    ),
}

# Public alias used by metadata and taxonomy contracts. The active production
# taxonomy is V3; V2 remains available only for reproducible comparison.
THEME_PATTERNS = THEME_PATTERNS_V3

FALLBACK_THEMES = ("other", "uncertain")
ALL_THEMES = tuple(THEME_PATTERNS) + FALLBACK_THEMES
CLASSIFICATION_METHOD = "deterministic_rule_based_multilabel"
POSITIVE_SHORT_PATTERN = (
    r"\b(recomendo|[oó]tim\w*|excelente|perfeit\w*|tudo correto|tudo certo|"
    r"muito bom|parab[eé]ns)\b"
)


def normalized_review_text(column: str = "review_text") -> Column:
    return F.trim(
        F.lower(
            F.regexp_replace(
                F.coalesce(F.col(column), F.lit("")), r"[^\p{L}]+", " "
            )
        )
    )


def complaint_theme_array(
    column: str = "review_text", classifier_version: str = "rules-pt-v3"
) -> Column:
    """Assign one or more themes, always including a deterministic fallback."""
    normalized = normalized_review_text(column)
    token_count = F.size(
        F.filter(
            F.split(normalized, r"\s+"), lambda token: F.length(token) > 1
        )
    )
    patterns = (
        THEME_PATTERNS_V2
        if classifier_version == "rules-pt-v2"
        else THEME_PATTERNS_V3
    )
    positive_short = (
        F.lit(False)
        if classifier_version == "rules-pt-v2"
        else normalized.rlike(POSITIVE_SHORT_PATTERN)
    )
    candidates = [
        F.when(normalized.rlike(pattern), F.lit(theme))
        for theme, pattern in patterns.items()
    ]
    matched = F.filter(F.array(*candidates), lambda value: value.isNotNull())
    return (
        F.when(F.size(matched) > 0, matched)
        .when(positive_short, F.array(F.lit("other")))
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
            complaint_theme_array(
                classifier_version=classifier_version
            ).alias("themes"),
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
