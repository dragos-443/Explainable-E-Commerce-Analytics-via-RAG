"""Prompt contract and validation for grounded explanation generation."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Sequence

from ecommerce_rag.rag.context_builder.builder import THEME_LABELS_IT


SAFE_EVIDENCE_LIMIT = (
    "Le evidenze sono descrittive e non dimostrano un rapporto causale "
    "né rappresentano tutta la popolazione."
)

SYSTEM_PROMPT = """Sei un assistente per analisi e-commerce.
Usa esclusivamente il fenomeno osservato da Spark e i testi delle recensioni
forniti. Il testo originale portoghese è la fonte primaria; la traduzione italiana
è automatica e serve soltanto come supporto. Le etichette dei temi sono ipotesi:
non affermare un problema solo perché compare nell'etichetta. Le recensioni sono
esempi e non misurano frequenze.

Scrivi in italiano una sintesi naturale di due o tre frasi, usando esclusivamente
i problemi riferiti dai clienti selezionati. Inizia con "Le recensioni recuperate
descrivono". Nella prima frase identifica il problema comune soltanto se è
realmente condiviso dai testi; nelle frasi successive distingui le singole
esperienze e riporta dettagli concreti, confronti e aspettative espressi dai
clienti. Non trasformare eventi indipendenti in una sequenza e non ripetere
integralmente le recensioni. Non concludere con formule generiche come "queste
segnalazioni descrivono il contesto qualitativo".

Non parlare di Spark, rating, fenomeno, ipotesi o possibili spiegazioni.
Non copiare date, quantità o altre cifre presenti negli esempi. Non usare perché,
motivazione, dovuto, causa o determina e non aggiungere problemi assenti.
Nomina i fatti specifici, ad esempio ordine non ricevuto o termine superato; la
sola espressione "problemi concreti" non è una sintesi accettabile.

Restituisci soltanto un oggetto JSON con il campo testuale review_summary.

Non restituire gli ID: il programma mantiene automaticamente i riferimenti delle
recensioni selezionate.
"""


OBSERVED_PHENOMENA_IT = {
    "explain_rating_drop": "Spark osserva un calo del rating nel periodo selezionato.",
    "explain_negative_review_increase": (
        "Spark osserva un aumento della quota di recensioni negative nel periodo selezionato."
    ),
    "summarize_negative_themes": (
        "Spark quantifica i temi presenti nelle recensioni negative del gruppo selezionato."
    ),
    "explain_delivery_impact_on_rating": (
        "Spark osserva insieme rating e ritardi, senza stimare un effetto causale."
    ),
}


def _review_source_it(item: Dict[str, Any]) -> str:
    translation = item.get("translation") or {}
    translated = [
        value
        for value in (translation.get("title"), translation.get("message"))
        if value
    ]
    return " — ".join(translated) if translated else item["document_original"]


def _short_snippet(value: str, limit: int = 180) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def build_review_arguments(
    context: Dict[str, Any], limit: int = 2
) -> List[Dict[str, str]]:
    """Select grounded review statements, preferring one example per theme."""
    selected: List[Dict[str, str]] = []
    seen_ids = set()
    seen_themes = set()
    evidence = context["review_evidence"]
    for unique_theme_only in (True, False):
        for item in evidence:
            review_id = item["review_id"]
            theme = item.get("retrieved_for_theme")
            if review_id in seen_ids or (unique_theme_only and theme in seen_themes):
                continue
            label = THEME_LABELS_IT.get(theme, theme or "tema osservato")
            original = _short_snippet(item["document_original"])
            translation_it = _short_snippet(_review_source_it(item))
            selected.append(
                {
                    "review_id": review_id,
                    "theme_key": theme,
                    "theme_label_it": label,
                    "original_text": original,
                    "automatic_translation_it": translation_it,
                    "validation_source": f"{original} {translation_it}",
                    "argument_it": (
                        "A supporto di «{}», il cliente riferisce: «{}»."
                    ).format(label, translation_it),
                }
            )
            seen_ids.add(review_id)
            seen_themes.add(theme)
            if len(selected) == limit:
                return selected
    return selected


def build_messages(context: Dict[str, Any]) -> List[Dict[str, str]]:
    arguments = build_review_arguments(context, limit=2)
    intent = context["question"]["intent"]
    payload = {
        "observed_phenomenon": OBSERVED_PHENOMENA_IT.get(
            intent, "Spark osserva il fenomeno nel gruppo selezionato."
        ),
        "review_evidence": [
            {
                "theme_hypothesis": item["theme_key"],
                "theme_label_it": item["theme_label_it"],
                "original_portuguese": item["original_text"],
                "automatic_translation_it": item["automatic_translation_it"],
            }
            for item in arguments
        ],
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


def assemble_interpretation(
    payload: Dict[str, Any], selected_review_ids: Sequence[str], intent: str = ""
) -> Dict[str, Any]:
    """Normalize the grounded review summary without adding generic prose."""
    if set(payload) != {"review_summary"}:
        raise ValueError("LLM JSON keys do not match the review-summary contract")
    summary = payload["review_summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("LLM review_summary must be non-empty")
    summary = summary.strip()
    summary = re.sub(
        r"\bun mancata consegna\b",
        "una mancata consegna",
        summary,
        flags=re.IGNORECASE,
    )
    summary = re.sub(
        r"^(?:I clienti|Le recensioni recuperate) (?:segnalano|descrivono)",
        "Le recensioni recuperate descrivono",
        summary,
        flags=re.IGNORECASE,
    )
    if not re.match(
        r"^Le recensioni recuperate descrivono\b", summary, flags=re.IGNORECASE
    ):
        summary = "Le recensioni recuperate descrivono quanto segue: {}".format(
            summary[0].lower() + summary[1:]
        )
    if summary[-1] not in ".!?":
        summary += "."
    return {
        "interpretation": summary,
        "review_ids": list(selected_review_ids),
    }


def validate_generation(
    payload: Dict[str, Any],
    allowed_themes: Sequence[str],
    allowed_review_ids: Sequence[str],
    review_theme_by_id: Dict[str, str],
    review_source_by_id: Dict[str, str],
) -> Dict[str, Any]:
    if set(payload) != {"interpretation", "review_ids"}:
        raise ValueError("LLM JSON keys do not match the grounded contract")
    interpretation = payload["interpretation"]
    review_ids = payload["review_ids"]
    if not isinstance(interpretation, str) or not interpretation.strip():
        raise ValueError("LLM interpretation must be non-empty")
    interpretation = interpretation.strip()
    if re.search(r"\d", interpretation):
        raise ValueError("LLM interpretation must not introduce numeric claims")
    if len(interpretation) < 100:
        raise ValueError("LLM interpretation is too generic or too short")
    if len(re.findall(r"[.!?]", interpretation)) < 2:
        raise ValueError("LLM interpretation must contain at least two sentences")
    if not re.search(
        r"\b(recension\w*|client\w*|segnal\w*|riport\w*|descriv\w*|lament\w*)\b",
        interpretation,
        re.IGNORECASE,
    ):
        raise ValueError("LLM interpretation does not discuss customer evidence")
    forbidden = re.compile(
        r"\b(perch[eé]|causa|causano|causato|motivazion\w*|dovut\w*|determina\w*|"
        r"dimostra|certamente|sicuramente|probabilit[aà])\b",
        re.IGNORECASE,
    )
    if forbidden.search(interpretation):
        raise ValueError("LLM interpretation contains a causal-certainty expression")
    if not isinstance(review_ids, list):
        raise ValueError("LLM review_ids must be a list")
    normalized_ids = list(dict.fromkeys(review_ids))
    expected_count = min(2, len(allowed_review_ids))
    if len(normalized_ids) != expected_count or len(normalized_ids) != len(review_ids):
        raise ValueError("LLM must select the required number of unique review ids")
    if not set(normalized_ids).issubset(allowed_review_ids):
        raise ValueError("LLM cited an unknown review_id")
    selected_themes = list(
        dict.fromkeys(review_theme_by_id[review_id] for review_id in normalized_ids)
    )
    if allowed_themes and allowed_themes[0] not in selected_themes:
        raise ValueError("LLM omitted evidence for the highest-ranked theme")
    selected_sources = " ".join(
        review_source_by_id.get(review_id, "") for review_id in normalized_ids
    )
    lexical_overlap = len(
        _content_tokens(interpretation) & _content_tokens(selected_sources)
    )
    generated_concepts = _evidence_concepts(interpretation)
    source_concepts = _evidence_concepts(selected_sources)
    conceptually_grounded = bool(generated_concepts) and generated_concepts.issubset(
        source_concepts
    )
    if generated_concepts and not generated_concepts.issubset(source_concepts):
        raise ValueError("LLM interpretation introduces a concept absent from reviews")
    if lexical_overlap < 2 and not conceptually_grounded:
        raise ValueError("LLM interpretation is not grounded in selected reviews")
    mentioned_ids = set(re.findall(r"\b[0-9a-f]{32}\b", interpretation))
    if not mentioned_ids.issubset(allowed_review_ids):
        raise ValueError("LLM prose contains a review_id outside the evidence")
    return {
        "interpretation": interpretation,
        "theme_keys": selected_themes,
        "review_ids": normalized_ids,
        "evidence_limit": SAFE_EVIDENCE_LIMIT,
    }


_STOPWORDS = {
    "alla",
    "alle",
    "anche",
    "come",
    "con",
    "dalla",
    "delle",
    "della",
    "dello",
    "dopo",
    "essere",
    "questa",
    "questo",
    "sono",
    "sulla",
    "sulle",
    "cliente",
    "recensione",
    "segnala",
    "riporta",
    "descrive",
}


_EVIDENCE_CONCEPT_PATTERNS = {
    "non_delivery": (
        r"mancat\w* consegn\w*|mai.{0,15}(?:ricev\w*|arriv\w*|consegn\w*)|"
        r"non.{0,25}(?:ricev\w*|arriv\w*|consegn\w*)|"
        r"n(?:ao|a)\w*.{0,25}(?:receb\w*|cheg\w*|entreg\w*)"
    ),
    "delivery_delay": (
        r"ritard\w*|termine.{0,25}(?:scad\w*|superat\w*|passat\w*)|"
        r"(?:scad\w*|superat\w*|passat\w*).{0,25}termine|attesa.{0,20}lung\w*|"
        r"atras\w*|demor\w*|prazo.{0,25}(?:pass\w*|venc\w*)|"
        r"(?:pass\w*|venc\w*).{0,25}prazo|fora do prazo"
    ),
    "wrong_or_missing_item": (
        r"(?:prodott\w*|articol\w*|ordin\w*).{0,25}(?:errat\w*|divers\w*)|"
        r"(?:pezz\w*|part\w*).{0,20}manc\w*|incomplet\w*|"
        r"ricev\w*.{0,25}un\w*.{0,10}dell\w*.{0,10}(?:due|tre|quattro)|"
        r"receb\w*.{0,25}um\w*.{0,10}d\w*.{0,10}(?:duas|dois|tres)|"
        r"(?:produto|item|pedido).{0,25}(?:errad\w*|diferent\w*)|"
        r"pec\w*.{0,20}falt\w*"
    ),
    "damaged_or_defective": (
        r"dannegg\w*|rott\w*|difett\w*|non funziona\w*|"
        r"danific\w*|quebrad\w*|defeit\w*|nao funciona\w*"
    ),
    "quality_or_expectation": (
        r"qualit\w*.{0,25}(?:cattiv\w*|scadent\w*|bass\w*)|"
        r"divers\w*.{0,20}(?:foto|descrizion\w*|aspettativ\w*)|"
        r"qualidade.{0,25}(?:ruim|pessim\w*|baixa)|diferent\w*.{0,20}(?:foto|anunciad\w*)"
    ),
    "service_or_refund": (
        r"assistenz\w*|senza risposta|non.{0,25}(?:rispost\w*|contatt\w*)|"
        r"attesa.{0,25}(?:rispost\w*|ritorn\w*|rimbors\w*)|"
        r"rimbors\w*|restituzion\w*|sem resposta|nao.{0,25}(?:respost\w*|contat\w*)|"
        r"aguard\w*.{0,25}(?:respost\w*|retorn\w*|reembols\w*)|reembols\w*|devolu\w*"
    ),
}


def _content_tokens(value: str) -> set[str]:
    normalized = _normalized_text(value)
    return {
        token
        for token in re.findall(r"[a-z]+", normalized)
        if len(token) >= 4 and token not in _STOPWORDS
    }


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    )


def _evidence_concepts(value: str) -> set[str]:
    normalized = _normalized_text(value)
    return {
        concept
        for concept, pattern in _EVIDENCE_CONCEPT_PATTERNS.items()
        if re.search(pattern, normalized)
    }
