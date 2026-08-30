"""Build and render a grounded context from Spark and RAG evidence."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


METRIC_LABELS = {
    "average_rating": "Rating medio",
    "negative_review_rate": "Quota di recensioni negative",
    "late_delivery_rate": "Quota di consegne tardive",
    "average_delivery_delay": "Ritardo medio delle consegne tardive",
    "order_volume": "Numero di ordini",
    "average_order_value": "Valore medio dell'ordine",
}


def _percentage(value: Optional[float]) -> str:
    return "n.d." if value is None else "{:.2f}%".format(value * 100)


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n.d."
    if isinstance(value, int):
        return str(value)
    return ("{:.%df}" % digits).format(value)


def _metric_line(metric: str, payload: Dict[str, Any]) -> str:
    label = METRIC_LABELS.get(metric, metric)
    percentage_metric = metric in {"negative_review_rate", "late_delivery_rate"}
    formatter = _percentage if percentage_metric else _number
    line = "{}: valore corrente {}".format(label, formatter(payload.get("value")))
    if payload.get("baseline_value") is not None:
        line += ", baseline {}".format(formatter(payload["baseline_value"]))
    if payload.get("change") is not None:
        change = (
            "{:+.2f} punti percentuali".format(payload["change"] * 100)
            if percentage_metric
            else "{:+.4f}".format(payload["change"])
        )
        line += ", variazione {}".format(change)
    denominator = payload.get("denominator")
    if denominator is not None:
        line += " (denominatore: {})".format(denominator)
    if payload.get("comparison_note"):
        line += " Il valore di baseline copre una finestra diversa e non viene usato per calcolare una variazione diretta"
    return line + "."


def _theme_line(item: Dict[str, Any], denominator: int) -> str:
    line = "{}: {} menzioni su {} ({})".format(
        item["theme"],
        item["current_mentions"],
        denominator,
        _percentage(item["current_mention_rate"]),
    )
    if item.get("baseline_mentions") is not None:
        line += ", baseline {} ({})".format(
            item["baseline_mentions"],
            _percentage(item["baseline_mention_rate"]),
        )
    if item.get("mention_rate_change") is not None:
        line += ", variazione {:+.2f} punti percentuali".format(
            item["mention_rate_change"] * 100
        )
    return line + "."


def build_grounded_context(
    question: Dict[str, Any],
    analytics: Dict[str, Any],
    theme_evidence: Dict[str, Any],
    review_evidence: List[Dict[str, Any]],
    max_ranked_themes: int = 3,
) -> Dict[str, Any]:
    metric = question["metric"]
    metrics = analytics["metrics"]
    selected_metrics = {}
    for key in (metric, "average_rating", "negative_review_rate", "late_delivery_rate"):
        if key in metrics and key not in selected_metrics:
            selected_metrics[key] = metrics[key]
    ranked = theme_evidence["ranked_hypotheses"][:max_ranked_themes]
    population = theme_evidence["current_population"]
    observations = [
        _metric_line(key, value)
        for key, value in selected_metrics.items()
        if isinstance(value, dict) and "value" in value
    ]
    observations.append(
        "Copertura testuale complessiva: {} recensioni su {} ({}).".format(
            population["text_reviews"],
            population["total_reviews"],
            _percentage(population["overall_text_coverage"]),
        )
    )
    observations.append(
        "Copertura testuale delle recensioni negative: {} su {} ({}); questo è il denominatore dei complaint theme.".format(
            population["negative_text_reviews"],
            population["negative_reviews"],
            _percentage(population["negative_text_coverage"]),
        )
    )
    observations.extend(
        _theme_line(item, population["negative_text_reviews"]) for item in ranked
    )
    return {
        "schema_version": "1.0",
        "question": question,
        "structured_analytics": analytics,
        "theme_prevalence": theme_evidence,
        "ranked_theme_evidence": ranked,
        "text_review_coverage": {
            key: population[key]
            for key in (
                "total_reviews",
                "text_reviews",
                "overall_text_coverage",
                "negative_reviews",
                "negative_text_reviews",
                "negative_text_coverage",
            )
        },
        "review_evidence": review_evidence,
        "observation_lines_it": observations,
        "grounding_rules": {
            "metrics_source": "Spark structured analytics only",
            "theme_prevalence_source": "Spark aggregation over the complete eligible group",
            "review_role": "representative top-k examples, not frequency estimates",
            "causality": "themes are ranked hypotheses, not demonstrated causes",
            "authoritative_review_text": "original",
        },
    }


def render_answer_it(
    context: Dict[str, Any],
    generation: Optional[Dict[str, Any]] = None,
    insufficient_reason: Optional[str] = None,
) -> str:
    lines = ["Osservazioni quantitative:"]
    lines.extend("- " + item for item in context["observation_lines_it"])
    if insufficient_reason:
        lines.extend(
            [
                "",
                "Evidenza insufficiente:",
                insufficient_reason,
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Interpretazione grounded:",
                generation["interpretation"],
                "Temi selezionati dall'LLM: {}.".format(
                    ", ".join(generation["theme_keys"])
                ),
                "Review citate dall'LLM: {}.".format(
                    ", ".join("[{}]".format(value) for value in generation["review_ids"])
                ),
                "",
                "Limite interpretativo:",
                generation["evidence_limit"],
            ]
        )
    lines.extend(["", "Recensioni recuperate:"])
    if not context["review_evidence"]:
        lines.append("- Nessuna recensione compatibile con i filtri.")
    for item in context["review_evidence"]:
        translation = item.get("translation") or {}
        translated_parts = [
            value for value in (translation.get("title"), translation.get("message")) if value
        ]
        lines.append(
            "- [{}] Originale: {}".format(
                item["review_id"], item["document_original"].replace("\n", " ")
            )
        )
        if translated_parts:
            lines.append(
                "  Traduzione automatica: {}".format(" — ".join(translated_parts))
            )
        elif translation:
            lines.append("  Traduzione automatica non disponibile; fa fede l'originale.")
    lines.extend(
        [
            "",
            "Nota: le prevalenze derivano dall'intero gruppo analizzato da Spark; le recensioni mostrate sono soltanto esempi top-k recuperati da Chroma.",
        ]
    )
    return "\n".join(lines)
