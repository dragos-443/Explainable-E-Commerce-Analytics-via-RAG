"""Controlled interpretation of Italian analytics questions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional, Tuple

from ecommerce_rag.analytics.engine import AnalyticsFilters


SUPPORTED_INTENTS = (
    "explain_rating_drop",
    "explain_negative_review_increase",
    "analyze_low_rating_complaints",
    "explain_delivery_impact_on_rating",
)

SUPPORTED_REQUESTED_THEMES = (
    "non_delivery",
    "delivery_delay",
    "wrong_or_missing_item",
    "damaged_or_defective",
    "quality_or_expectation",
    "service_or_refund",
)

THEME_ALIASES: Dict[str, Tuple[str, ...]] = {
    "non_delivery": (
        "mancata consegna",
        "non consegnato",
        "non ricevuto",
        "mai arrivato",
    ),
    "delivery_delay": (
        "ritardo nella consegna",
        "ritardi nella consegna",
        "consegna in ritardo",
        "consegne in ritardo",
        "ritardo",
        "ritardi",
    ),
    "wrong_or_missing_item": (
        "articolo errato",
        "articoli errati",
        "prodotto errato",
        "prodotti errati",
        "articolo sbagliato",
        "prodotto sbagliato",
        "articolo mancante",
        "articoli mancanti",
        "parti mancanti",
        "prodotto incompleto",
        "prodotti incompleti",
    ),
    "damaged_or_defective": (
        "danneggiato",
        "danneggiati",
        "danneggiate",
        "difetto",
        "difetti",
        "difettoso",
        "difettosi",
        "rotto",
        "rotti",
        "non funzionante",
        "non funzionanti",
    ),
    "quality_or_expectation": (
        "qualita",
        "aspettative",
        "diverso dalla descrizione",
        "diversi dalla descrizione",
        "diverso dalla foto",
        "diversi dalla foto",
    ),
    "service_or_refund": (
        "assistenza",
        "servizio clienti",
        "rimborso",
        "rimborsi",
        "reso",
        "resi",
    ),
}

CATEGORY_ALIASES: Dict[str, str] = {
    "office furniture": "office_furniture",
    "mobili per ufficio": "office_furniture",
    "arredamento per ufficio": "office_furniture",
    "arredamento ufficio": "office_furniture",
    "sport e tempo libero": "sports_leisure",
    "salute e bellezza": "health_beauty",
    "informatica e accessori": "computers_accessories",
    "casa e giardino": "housewares",
    "libri": "books_general_interest",
    "piccoli elettrodomestici": "small_appliances",
    "elettronica": "electronics",
    "profumeria": "perfumery",
    "profumi": "perfumery",
    "letto bagno e tavola": "bed_bath_table",
    "letto, bagno e tavola": "bed_bath_table",
}

ITALIAN_MONTHS = {
    "gennaio": 1,
    "febbraio": 2,
    "marzo": 3,
    "aprile": 4,
    "maggio": 5,
    "giugno": 6,
    "luglio": 7,
    "agosto": 8,
    "settembre": 9,
    "ottobre": 10,
    "novembre": 11,
    "dicembre": 12,
}


def _normalized(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _extract_period(question: str) -> Tuple[Optional[str], Optional[str]]:
    explicit = re.findall(r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])\b", question)
    if explicit:
        return explicit[0], explicit[1] if len(explicit) > 1 else explicit[0]
    normalized = _normalized(question)
    matches = re.findall(
        r"\b(" + "|".join(ITALIAN_MONTHS) + r")\s+((?:19|20)\d{2})\b",
        normalized,
    )
    if not matches:
        return None, None
    periods = [
        "{}-{:02d}".format(year, ITALIAN_MONTHS[month])
        for month, year in matches
    ]
    return periods[0], periods[1] if len(periods) > 1 else periods[0]


def _extract_category(question: str) -> Optional[str]:
    normalized = _normalized(question).replace("_", " ")
    for alias, category in sorted(
        CATEGORY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if alias in normalized:
            return category
    return None


def _extract_requested_theme(question: str) -> Optional[str]:
    normalized = _normalized(question)
    matches = []
    for theme, aliases in THEME_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                matches.append((len(alias), theme))
    if not matches:
        return None
    return max(matches)[1]


def _intent_and_metric(
    question: str, requested_theme: Optional[str] = None
) -> Tuple[str, str]:
    normalized = _normalized(question)
    rating = any(token in normalized for token in ("rating", "valutaz", "punteggio"))
    delivery = any(
        token in normalized for token in ("consegna", "ritard", "recapit")
    )
    negative = "recension" in normalized and any(
        token in normalized for token in ("negativ", "bassa", "basso")
    )
    increase = any(
        token in normalized for token in ("aument", "increment", "cresc")
    )
    complaints = any(
        token in normalized for token in ("problem", "lament", "reclam", "motivi")
    )
    if any(
        token in normalized
        for token in (
            "volume ordini",
            "volume degli ordini",
            "numero di ordini",
            "numero degli ordini",
        )
    ):
        return "descriptive_only", "order_volume"
    if any(
        token in normalized
        for token in ("valore medio", "valore degli ordini", "valore medio degli ordini")
    ):
        return "descriptive_only", "average_order_value"
    if delivery and rating:
        return "explain_delivery_impact_on_rating", "late_delivery_rate"
    if negative and increase:
        return "explain_negative_review_increase", "negative_review_rate"
    if rating:
        return "explain_rating_drop", "average_rating"
    if negative or complaints or requested_theme:
        return "analyze_low_rating_complaints", "negative_review_rate"
    return "unsupported", "unknown"


@dataclass(frozen=True)
class InterpretedQuestion:
    question_original: str
    question_language: str
    intent: str
    metric: str
    requested_theme: Optional[str]
    category: Optional[str]
    customer_state: Optional[str]
    start_month: Optional[str]
    end_month: Optional[str]
    interpretation_method: str = "controlled_rules_v2"

    def as_dict(self) -> dict:
        return asdict(self)

    def analytics_filters(self) -> AnalyticsFilters:
        return AnalyticsFilters(
            product_category=self.category,
            customer_state=self.customer_state,
            start_month=self.start_month,
            end_month=self.end_month,
        ).validated()


def interpret_question(
    question: str,
    *,
    intent: Optional[str] = None,
    metric: Optional[str] = None,
    requested_theme: Optional[str] = None,
    category: Optional[str] = None,
    customer_state: Optional[str] = None,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
) -> InterpretedQuestion:
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    inferred_theme = _extract_requested_theme(question)
    selected_theme = requested_theme or inferred_theme
    if selected_theme is not None and selected_theme not in SUPPORTED_REQUESTED_THEMES:
        raise ValueError("Unknown requested complaint theme: {}".format(selected_theme))
    inferred_intent, inferred_metric = _intent_and_metric(question, selected_theme)
    inferred_start, inferred_end = _extract_period(question)
    interpreted = InterpretedQuestion(
        question_original=question.strip(),
        question_language="it",
        intent=intent or inferred_intent,
        metric=metric or inferred_metric,
        requested_theme=selected_theme,
        category=category or _extract_category(question),
        customer_state=customer_state,
        start_month=start_month or inferred_start,
        end_month=end_month or (inferred_end if not start_month else start_month),
    )
    if interpreted.intent not in SUPPORTED_INTENTS + ("descriptive_only", "unsupported"):
        raise ValueError("Unknown controlled intent: {}".format(interpreted.intent))
    filters = interpreted.analytics_filters()
    return replace(
        interpreted,
        category=filters.product_category,
        customer_state=filters.customer_state,
        start_month=filters.start_month,
        end_month=filters.end_month,
    )
