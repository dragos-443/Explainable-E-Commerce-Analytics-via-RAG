"""End-to-end integration of Spark analytics, RAG evidence and grounded LLM synthesis."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ecommerce_rag.analytics.engine import AnalyticsEngine
from ecommerce_rag.analytics.theme_evidence import ThemeEvidenceEngine
from ecommerce_rag.rag.context_builder.builder import (
    THEME_LABELS_IT,
    build_grounded_context,
    render_answer_it,
)
from ecommerce_rag.rag.prompting.grounded import (
    assemble_interpretation,
    build_messages,
    build_review_arguments,
    extract_json,
    validate_generation,
)
from ecommerce_rag.rag.question_interpreter import InterpretedQuestion
from ecommerce_rag.rag.retrieval.service import RetrievalFilters, ReviewRetriever


THEME_RETRIEVAL_DESCRIPTIONS_IT = {
    "non_delivery": "ordine, prodotto o merce non ricevuti, non arrivati o mai consegnati",
    "delivery_delay": "consegna in ritardo, termine superato o attesa troppo lunga",
    "wrong_or_missing_item": "prodotto errato, diverso, incompleto o con parti mancanti",
    "damaged_or_defective": "prodotto danneggiato, rotto, difettoso o non funzionante",
    "quality_or_expectation": "qualità scadente o prodotto diverso da foto, descrizione e aspettative",
    "service_or_refund": "assistenza senza risposta, problema non risolto, reso o rimborso",
}


def focus_theme_evidence(
    question: InterpretedQuestion, theme_evidence: Dict[str, Any]
) -> Dict[str, Any]:
    """Restrict ranked hypotheses when the user explicitly asks for one theme."""
    if not question.requested_theme:
        return theme_evidence
    requested = [
        item
        for item in theme_evidence["ranked_hypotheses"]
        if item["theme"] == question.requested_theme
    ]
    focused = dict(theme_evidence)
    focused["ranked_hypotheses"] = requested
    focused["requested_theme"] = question.requested_theme
    focused["ranking_note"] = (
        "The user explicitly requested one complaint theme. Its support is computed "
        "by Spark over the complete eligible group; Chroma only selects examples."
    )
    return focused


def theme_retrieval_query(question: str, theme: str) -> str:
    """Focus semantic retrieval and reranking on the current theme hypothesis."""
    description = THEME_RETRIEVAL_DESCRIPTIONS_IT.get(theme)
    if not description:
        return question
    return f"{question} Cerca recensioni che descrivono: {description}."


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
        if question.requested_theme:
            return (
                "Il tema richiesto ({}) non ha menzioni nelle recensioni negative "
                "con testo selezionate."
            ).format(THEME_LABELS_IT.get(question.requested_theme, question.requested_theme))
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
        focused_query = theme_retrieval_query(question.question_original, theme["theme"])
        filters = RetrievalFilters(
            product_category=question.category,
            customer_state=question.customer_state,
            start_month=question.start_month,
            end_month=question.end_month,
            max_review_score=2,
            theme=theme["theme"],
        )
        results = retriever.retrieve(
            focused_query,
            top_k=evidence_per_theme,
            filters=filters,
            translate=translate,
        )
        for item in results:
            if item["review_id"] in seen:
                continue
            seen.add(item["review_id"])
            item["retrieved_for_theme"] = theme["theme"]
            item["retrieval_query"] = focused_query
            evidence.append(item)
    return evidence


def _safe_fallback(context: Dict[str, Any], error: str) -> Dict[str, Any]:
    arguments = build_review_arguments(context)
    themes = list(
        dict.fromkeys(
            item["theme_key"] for item in arguments if item.get("theme_key")
        )
    )
    review_ids = [item["review_id"] for item in arguments]
    details = [
        item["automatic_translation_it"].strip().rstrip(".!?;:")
        for item in arguments
        if item.get("automatic_translation_it")
    ]
    if len(details) >= 2:
        fallback_interpretation = (
            "Le recensioni recuperate descrivono esperienze distinte. "
            "Una recensione riferisce: {}; un'altra segnala: {}."
        ).format(details[0], details[1])
    elif details:
        fallback_interpretation = (
            "Le recensioni recuperate descrivono un'esperienza specifica. "
            "Il cliente riferisce: {}."
        ).format(details[0])
    else:
        fallback_interpretation = (
            "Le recensioni recuperate non contengono dettagli testuali sufficienti "
            "per produrre una sintesi più specifica."
        )
    return {
        "quantitative_summary": context["quantitative_summary_it"],
        "theme_summary": context["theme_summary_it"],
        "interpretation": fallback_interpretation,
        "theme_keys": themes,
        "review_ids": review_ids,
        "review_arguments": arguments,
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
        unload_generator = getattr(self.generator, "unload", None)
        if unload_generator is not None:
            unload_generator()
        filters = question.analytics_filters()
        analytics = self.analytics_engine.analyze(filters, query_id=query_id)
        all_themes = self.theme_engine.analyze(filters)
        themes = focus_theme_evidence(question, all_themes)
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
        context_theme_limit = 0 if question.intent in {
            "descriptive_only",
            "unsupported",
        } else (1 if question.requested_theme else themes_limit)
        context = build_grounded_context(
            question.as_dict(), analytics, themes, evidence, context_theme_limit
        )
        generation = None
        raw_generation = None
        generation_attempts = []
        if reason is None:
            release_models = getattr(self.retriever, "release_models", None)
            if release_models is not None:
                release_models()
            allowed_themes = [item["theme"] for item in context["ranked_theme_evidence"]]
            argument_candidates = build_review_arguments(context)
            allowed_ids = [item["review_id"] for item in argument_candidates]
            review_theme_by_id = {
                item["review_id"]: item["theme_key"] for item in argument_candidates
            }
            review_source_by_id = {
                item["review_id"]: item["validation_source"]
                for item in argument_candidates
            }
            arguments_by_id = {
                item["review_id"]: item for item in argument_candidates
            }
            messages = build_messages(context)
            last_error = None
            for attempt in range(1):
                try:
                    raw_generation = self.generator.generate(messages)
                    generation_attempts.append(raw_generation)
                    generation = validate_generation(
                        assemble_interpretation(
                            extract_json(raw_generation),
                            allowed_ids,
                            context["question"]["intent"],
                        ),
                        allowed_themes,
                        allowed_ids,
                        review_theme_by_id,
                        review_source_by_id,
                    )
                    generation["quantitative_summary"] = context[
                        "quantitative_summary_it"
                    ]
                    generation["theme_summary"] = context["theme_summary_it"]
                    generation["review_arguments"] = [
                        arguments_by_id[review_id]
                        for review_id in generation["review_ids"]
                    ]
                    generation["generation_status"] = "llm_generated_validated"
                    generation["fallback_reason"] = None
                    break
                except (ValueError, TypeError, RuntimeError) as exc:
                    last_error = str(exc)
            if generation is None:
                generation = _safe_fallback(context, last_error or "unknown error")
        answer = render_answer_it(context, generation, reason)
        llm_backend = {
            "provider": getattr(self.generator, "last_provider", None)
            or getattr(self.generator, "provider", None),
            "model": getattr(self.generator, "last_model", None)
            or getattr(self.generator, "model_name", None),
        }
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
            "llm_backend": llm_backend,
            "answer_language": "it",
            "answer_it": answer,
        }
