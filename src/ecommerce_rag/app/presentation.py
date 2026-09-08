"""Pure presentation helpers for the Phase 8 Streamlit application."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional

THEME_LABELS = {
    "non_delivery": "Ordine non ricevuto",
    "delivery_delay": "Ritardo nella consegna",
    "damaged_or_defective": "Prodotto danneggiato o difettoso",
    "wrong_or_missing_item": "Prodotto errato o incompleto",
    "quality_or_expectation": "Qualità inferiore alle aspettative",
    "service_or_refund": "Assistenza o rimborso",
    "other": "Altro",
    "uncertain": "Tema incerto",
}

CATEGORY_LABELS = {
    "office_furniture": "Mobili per ufficio",
    "sports_leisure": "Sport e tempo libero",
    "health_beauty": "Salute e bellezza",
    "computers_accessories": "Informatica e accessori",
    "housewares": "Casa e giardino",
    "books_general_interest": "Libri",
    "small_appliances": "Piccoli elettrodomestici",
    "electronics": "Elettronica",
    "perfumery": "Profumeria",
    "bed_bath_table": "Letto, bagno e tavola",
}

MONTH_LABELS = (
    "", "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
    "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre",
)

METRIC_SPECS = (
    ("average_rating", "Rating medio", "decimal"),
    ("negative_review_rate", "Recensioni negative", "percentage"),
    ("late_delivery_rate", "Consegne tardive", "percentage"),
    ("order_volume", "Numero di ordini", "integer"),
)


def query_identifier(question: str, filters: Mapping[str, Optional[str]]) -> str:
    payload = json.dumps(
        {"question": question.strip(), "filters": dict(filters)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return "streamlit-{}".format(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12])


def _month_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        year, month = value.split("-")
        return "{} {}".format(MONTH_LABELS[int(month)], year)
    except (ValueError, IndexError):
        return value


def analysis_scope(question: Mapping[str, Any]) -> str:
    """Describe interpreted filters with user-facing Italian labels."""
    category = question.get("category")
    parts = [
        CATEGORY_LABELS.get(category, category.replace("_", " ").capitalize())
        if category
        else "Tutte le categorie"
    ]
    requested_theme = question.get("requested_theme")
    if requested_theme:
        parts.append(THEME_LABELS.get(requested_theme, requested_theme))
    if question.get("customer_state"):
        parts.append("Stato: {}".format(question["customer_state"]))
    start = _month_label(question.get("start_month"))
    end = _month_label(question.get("end_month"))
    if start and end:
        parts.append(start if start == end else "Da {} a {}".format(start, end))
    elif start or end:
        parts.append(start or end)
    return " · ".join(parts)


def _value(value: Any, kind: str, *, signed: bool = False) -> str:
    if value is None:
        return "n.d."
    if kind == "percentage":
        return ("{:+.2f}" if signed else "{:.2f}").format(float(value) * 100) + "%"
    if kind == "integer":
        return "{:,}".format(int(value)).replace(",", ".")
    return ("{:+.3f}" if signed else "{:.3f}").format(float(value))


def metric_cards(result: Mapping[str, Any]) -> List[Dict[str, Optional[str]]]:
    metrics = result["analytics"]["metrics"]
    cards = []
    for key, label, kind in METRIC_SPECS:
        payload = metrics[key]
        change = payload.get("change")
        delta = _value(change, kind, signed=True) if change is not None else None
        cards.append(
            {
                "key": key,
                "label": label,
                "value": _value(payload.get("value"), kind),
                "delta": delta,
                "baseline": _value(payload.get("baseline_value"), kind),
            }
        )
    return cards


def comparison_rows(result: Mapping[str, Any]) -> List[Dict[str, str]]:
    return [
        {
            "Metrica": card["label"],
            "Periodo selezionato": card["value"],
            "Baseline precedente": card["baseline"],
            "Variazione": card["delta"] or "n.d.",
        }
        for card in metric_cards(result)
    ]


def theme_rows(result: Mapping[str, Any], limit: int = 5) -> List[Dict[str, str]]:
    rows = []
    has_baseline = bool(
        result.get("analytics", {}).get("baseline", {}).get("filters")
    )
    for item in result["theme_evidence"]["ranked_hypotheses"][:limit]:
        row = {
            "Tema": THEME_LABELS.get(item["theme"], item["theme"]),
            "Menzioni": str(item["current_mentions"]),
            "Quota nel gruppo": _value(
                item.get("current_mention_rate"), "percentage"
            ),
        }
        if has_baseline:
            row["Variazione vs baseline"] = _value(
                item.get("mention_rate_change"), "percentage", signed=True
            )
        rows.append(row)
    return rows


def rating_display(value: Any) -> str:
    """Return a compact review score suited to an expander heading."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return "Valutazione n.d."
    if not 1 <= score <= 5:
        return "Valutazione n.d."
    return "⭐ {}/5".format(score)


def review_view(item: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata") or {}
    translation = item.get("translation") or {}
    translated_parts = [
        value for value in (translation.get("title"), translation.get("message")) if value
    ]
    return {
        "review_id": item["review_id"],
        "original": item["document_original"],
        "language": metadata.get("review_language"),
        "translation": " — ".join(translated_parts) if translated_parts else None,
        "translation_status": translation.get("status", "not_requested"),
        "translation_error": translation.get("error"),
        "translation_glossary_version": translation.get(
            "translation_glossary_version"
        ),
        "glossary_corrections": translation.get("glossary_corrections") or [],
        "review_score": metadata.get("review_score"),
        "product_category": metadata.get("product_category"),
        "customer_state": metadata.get("customer_state"),
        "purchase_month": metadata.get("purchase_month"),
        "retrieved_for_theme": THEME_LABELS.get(
            item.get("retrieved_for_theme"), item.get("retrieved_for_theme")
        ),
    }
