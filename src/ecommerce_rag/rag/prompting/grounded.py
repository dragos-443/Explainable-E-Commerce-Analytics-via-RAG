"""Prompt contract and validation for grounded explanation generation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence


SYSTEM_PROMPT = """Sei un assistente per analisi e-commerce.
Usa esclusivamente il contesto fornito. Spark ha già calcolato tutte le metriche:
non calcolare, modificare o inventare numeri. I complaint theme sono ipotesi
ordinate per supporto osservato, non cause o probabilità causali. Le recensioni
top-k sono esempi e non misurano la prevalenza. Rispondi in italiano.

Restituisci esclusivamente JSON valido con questa struttura:
{"interpretation":"massimo tre frasi prudenti, senza cifre; deve iniziare con: Le evidenze sono compatibili con le ipotesi selezionate.",
 "theme_keys":["solo chiavi ammesse"],
 "review_ids":["solo identificatori ammessi"],
 "evidence_limit":"Le evidenze sono descrittive e non dimostrano un rapporto causale né rappresentano tutta la popolazione."}

Includi sempre il primo tema ammesso, perché è quello con il maggiore supporto
osservato da Spark. Non usare parole come perché, dovuto, causa o determina per
collegare i temi alla variazione del KPI.
"""


def build_messages(context: Dict[str, Any]) -> List[Dict[str, str]]:
    themes = context["ranked_theme_evidence"]
    reviews = context["review_evidence"]
    allowed_themes = [item["theme"] for item in themes]
    allowed_ids = [item["review_id"] for item in reviews]
    evidence = []
    for item in reviews:
        translation = item.get("translation") or {}
        evidence.append(
            {
                "review_id": item["review_id"],
                "retrieved_for_theme": item.get("retrieved_for_theme"),
                "original": item["document_original"],
                "automatic_translation_it": {
                    "title": translation.get("title"),
                    "message": translation.get("message"),
                    "status": translation.get("status"),
                },
            }
        )
    payload = {
        "question": context["question"]["question_original"],
        "intent": context["question"]["intent"],
        "observations_from_spark": context["observation_lines_it"],
        "allowed_theme_keys": allowed_themes,
        "allowed_review_ids": allowed_ids,
        "review_evidence": evidence,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def extract_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM output does not contain a JSON object")
    return json.loads(text[start : end + 1])


def validate_generation(
    payload: Dict[str, Any],
    allowed_themes: Sequence[str],
    allowed_review_ids: Sequence[str],
) -> Dict[str, Any]:
    required = {"interpretation", "theme_keys", "review_ids", "evidence_limit"}
    if set(payload) != required:
        raise ValueError("LLM JSON keys do not match the grounded contract")
    if not isinstance(payload["interpretation"], str) or not payload["interpretation"].strip():
        raise ValueError("LLM interpretation must be non-empty")
    if not isinstance(payload["evidence_limit"], str) or not payload["evidence_limit"].strip():
        raise ValueError("LLM evidence_limit must be non-empty")
    if re.search(r"\d", payload["interpretation"] + payload["evidence_limit"]):
        raise ValueError("LLM prose must not introduce numeric claims")
    normalized_interpretation = payload["interpretation"].strip()
    if not normalized_interpretation.lower().startswith(
        "le evidenze sono compatibili con le ipotesi selezionate"
    ):
        raise ValueError("LLM interpretation is missing the required hypothesis framing")
    forbidden = re.compile(
        r"\b(perch[eé]|causa|causano|causato|dovut\w*|determina\w*|"
        r"dimostra|certamente|sicuramente|probabilit[aà])\b",
        re.IGNORECASE,
    )
    if forbidden.search(payload["interpretation"]):
        raise ValueError("LLM interpretation contains a causal-certainty expression")
    themes = payload["theme_keys"]
    review_ids = payload["review_ids"]
    if not isinstance(themes, list) or not set(themes).issubset(allowed_themes):
        raise ValueError("LLM selected an unknown theme")
    if not isinstance(review_ids, list) or not set(review_ids).issubset(allowed_review_ids):
        raise ValueError("LLM cited an unknown review_id")
    if allowed_themes and not themes:
        raise ValueError("LLM did not select any supported theme")
    if allowed_themes and allowed_themes[0] not in themes:
        raise ValueError("LLM omitted the highest-ranked supported theme")
    if allowed_review_ids and not review_ids:
        raise ValueError("LLM did not cite any retrieved review")
    mentioned_ids = set(re.findall(r"\b[0-9a-f]{32}\b", payload["interpretation"]))
    if not mentioned_ids.issubset(allowed_review_ids):
        raise ValueError("LLM prose contains a review_id outside the evidence")
    required_limit = (
        "Le evidenze sono descrittive e non dimostrano un rapporto causale "
        "né rappresentano tutta la popolazione."
    )
    if payload["evidence_limit"].strip() != required_limit:
        raise ValueError("LLM evidence_limit does not match the safe grounding statement")
    return {
        "interpretation": payload["interpretation"].strip(),
        "theme_keys": list(dict.fromkeys(themes)),
        "review_ids": list(dict.fromkeys(review_ids)),
        "evidence_limit": payload["evidence_limit"].strip(),
    }
