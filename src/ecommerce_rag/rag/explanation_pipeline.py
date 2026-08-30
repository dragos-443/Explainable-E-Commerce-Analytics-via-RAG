"""End-to-end integration of Spark analytics, RAG evidence and grounded LLM synthesis."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ecommerce_rag.analytics.engine import AnalyticsEngine
from ecommerce_rag.analytics.theme_evidence import ThemeEvidenceEngine
from ecommerce_rag.rag.context_builder.builder import (
    build_grounded_context,
    render_answer_it,
)
from ecommerce_rag.rag.prompting.grounded import (
    build_messages,
    extract_json,
    validate_generation,
)
from ecommerce_rag.rag.question_interpreter import InterpretedQuestion
from ecommerce_rag.rag.retrieval.service import RetrievalFilters, ReviewRetriever


def insufficient_evidence_reason(
    question: InterpretedQuestion,
    analytics: Dict[str, Any],
    theme_evidence: Dict[str, Any],
) -> Optional[str]:
    if question.intent == "unsupported":
        return "La domanda non corrisponde a uno degli intent controllati disponibili."
    if question.intent == "descriptive_only":
        return (
            "La metrica richiesta è descrittiva e le recensioni non forniscono "
            "una base adeguata per spiegarne automaticamente la variazione."
        )
    if analytics["populations"]["reviews"] == 0:
        return "I filtri non selezionano recensioni su cui calcolare il fenomeno."
    if theme_evidence["current_population"]["negative_text_reviews"] == 0:
        return "Non sono disponibili recensioni negative con testo per i filtri scelti."
    if question.intent == "explain_delivery_impact_on_rating":
        return (
            "Le metriche disponibili descrivono rating e ritardi, ma non stimano "
            "un effetto causale dei ritardi sul rating."
        )
    if question.intent == "explain_rating_drop":
        change = analytics["metrics"]["rating_variation"]["value"]
        if change is None:
            return "Per verificare un calo del rating serve un periodo con baseline."
        if change >= 0:
            return "Nel periodo selezionato Spark non osserva un calo del rating rispetto alla baseline."
    if question.intent == "explain_negative_review_increase":
        change = analytics["metrics"]["negative_review_rate"]["change"]
        if change is None:
            return "Per verificare l'aumento delle recensioni negative serve un periodo con baseline."
        if change <= 0:
            return "Nel periodo selezionato Spark non osserva un aumento della quota negativa."
    if not theme_evidence["ranked_hypotheses"]:
        return "Non esistono complaint theme con supporto nel gruppo selezionato."
    return None


def retrieve_ranked_evidence(
    question: InterpretedQuestion,
    theme_evidence: Dict[str, Any],
    retriever: ReviewRetriever,
    *,
    themes_limit: int = 3,
    evidence_per_theme: int = 2,
    translate: bool = True,
) -> List[Dict[str, Any]]:
    ranked = theme_evidence["ranked_hypotheses"]
    if question.intent == "explain_delivery_impact_on_rating":
        ranked = sorted(
            ranked,
            key=lambda item: (item["theme"] != "delivery_delay", item["rank"]),
        )
    selected_themes = ranked[:themes_limit]
    seen = set()
    evidence = []
    for theme in selected_themes:
        filters = RetrievalFilters(
            product_category=question.category,
            customer_state=question.customer_state,
            start_month=question.start_month,
            end_month=question.end_month,
            max_review_score=2,
            theme=theme["theme"],
        )
        results = retriever.retrieve(
            question.question_original,
            top_k=evidence_per_theme,
            filters=filters,
            translate=translate,
        )
        for item in results:
            if item["review_id"] in seen:
                continue
            seen.add(item["review_id"])
            item["retrieved_for_theme"] = theme["theme"]
            evidence.append(item)
    return evidence


def _safe_fallback(context: Dict[str, Any], error: str) -> Dict[str, Any]:
    themes = [item["theme"] for item in context["ranked_theme_evidence"][:2]]
    review_ids = [item["review_id"] for item in context["review_evidence"][:3]]
    readable = ", ".join(themes) if themes else "i temi disponibili"
    return {
        "interpretation": (
            "I segnali più supportati riguardano {}. Sono ipotesi compatibili "
            "con le variazioni osservate e vanno lette insieme alle recensioni citate."
        ).format(readable),
        "theme_keys": themes,
        "review_ids": review_ids,
        "evidence_limit": (
            "Le evidenze sono descrittive e gli esempi recuperati non provano un rapporto causale."
        ),
        "generation_status": "validated_fallback",
        "fallback_reason": error,
    }


class ExplanationPipeline:
    def __init__(
        self,
        analytics_engine: AnalyticsEngine,
        theme_engine: ThemeEvidenceEngine,
        retriever: ReviewRetriever,
        generator,
    ):
        self.analytics_engine = analytics_engine
        self.theme_engine = theme_engine
        self.retriever = retriever
        self.generator = generator

    def run(
        self,
        question: InterpretedQuestion,
        *,
        query_id: str,
        themes_limit: int = 3,
        evidence_per_theme: int = 2,
        translate: bool = True,
    ) -> Dict[str, Any]:
        filters = question.analytics_filters()
        analytics = self.analytics_engine.analyze(filters, query_id=query_id)
        themes = self.theme_engine.analyze(filters)
        reason = insufficient_evidence_reason(question, analytics, themes)
        evidence = []
        if reason is None:
            evidence = retrieve_ranked_evidence(
                question,
                themes,
                self.retriever,
                themes_limit=themes_limit,
                evidence_per_theme=evidence_per_theme,
                translate=translate,
            )
            if not evidence:
                reason = "Chroma non ha restituito recensioni compatibili con temi e filtri."
        context_theme_limit = (
            0 if question.intent in {"descriptive_only", "unsupported"} else themes_limit
        )
        context = build_grounded_context(
            question.as_dict(), analytics, themes, evidence, context_theme_limit
        )
        generation = None
        raw_generation = None
        generation_attempts = []
        if reason is None:
            allowed_themes = [item["theme"] for item in context["ranked_theme_evidence"]]
            allowed_ids = [item["review_id"] for item in evidence]
            messages = build_messages(context)
            last_error = None
            for attempt in range(2):
                try:
                    raw_generation = self.generator.generate(messages)
                    generation_attempts.append(raw_generation)
                    generation = validate_generation(
                        extract_json(raw_generation), allowed_themes, allowed_ids
                    )
                    generation["generation_status"] = "llm_generated_validated"
                    generation["fallback_reason"] = None
                    break
                except (ValueError, TypeError, RuntimeError) as exc:
                    last_error = str(exc)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Il precedente JSON è stato rifiutato: {}. "
                                "Restituisci un nuovo JSON che rispetti esattamente "
                                "il contratto, senza testo aggiuntivo."
                            ).format(last_error),
                        }
                    )
            if generation is None:
                generation = _safe_fallback(context, last_error or "unknown error")
        answer = render_answer_it(context, generation, reason)
        return {
            "schema_version": "1.0",
            "query_id": query_id,
            "question": question.as_dict(),
            "analytics": analytics,
            "theme_evidence": themes,
            "context": context,
            "insufficient_evidence": {
                "is_insufficient": reason is not None,
                "reason": reason,
            },
            "generation": generation,
            "raw_llm_output": raw_generation,
            "llm_generation_attempts": generation_attempts,
            "answer_language": "it",
            "answer_it": answer,
        }
