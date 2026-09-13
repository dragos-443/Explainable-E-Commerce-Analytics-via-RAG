"""Data-driven demo cases and executable verification rules for Phase 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    title: str
    kind: str
    question: str
    expected_intent: str
    expected_theme: Optional[str] = None
    category: Optional[str] = None
    start_month: Optional[str] = None
    end_month: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


DEMO_CASES: Tuple[DemoCase, ...] = (
    DemoCase(
        case_id="logistics_march_2018",
        title="Calo del rating e problemi logistici a marzo 2018",
        kind="logistics",
        question="Perché il rating è diminuito a marzo 2018?",
        expected_intent="explain_rating_drop",
        start_month="2018-03",
        end_month="2018-03",
    ),
    DemoCase(
        case_id="office_furniture_product_issues",
        title="Problemi nelle recensioni negative dei mobili per ufficio",
        kind="product_issues",
        question=(
            "Quali sono i principali problemi riportati nelle recensioni "
            "negative dei mobili per ufficio?"
        ),
        expected_intent="analyze_low_rating_complaints",
        category="office_furniture",
    ),
    DemoCase(
        case_id="overall_complaint_analysis",
        title="Principali problemi riportati dai clienti",
        kind="review_analysis",
        question="Quali sono i principali problemi riportati nelle recensioni negative?",
        expected_intent="analyze_low_rating_complaints",
    ),
    DemoCase(
        case_id="order_volume_insufficient",
        title="Volume ordini ed evidenza qualitativa insufficiente",
        kind="insufficient_evidence",
        question="Perché è cambiato il volume degli ordini a marzo 2018?",
        expected_intent="descriptive_only",
        start_month="2018-03",
        end_month="2018-03",
    ),
)


THEMATIC_EXAMPLE_CASES: Tuple[DemoCase, ...] = (
    DemoCase(
        case_id="bed_bath_table_quality",
        title="Qualità dei prodotti per letto, bagno e tavola",
        kind="thematic_focus",
        question=(
            "Quali problemi di qualità emergono nelle recensioni negative "
            "dei prodotti per letto, bagno e tavola?"
        ),
        expected_intent="analyze_low_rating_complaints",
        expected_theme="quality_or_expectation",
        category="bed_bath_table",
    ),
    DemoCase(
        case_id="electronics_defects",
        title="Prodotti elettronici danneggiati o difettosi",
        kind="thematic_focus",
        question=(
            "Quali difetti vengono segnalati nelle recensioni negative "
            "dei prodotti di elettronica?"
        ),
        expected_intent="analyze_low_rating_complaints",
        expected_theme="damaged_or_defective",
        category="electronics",
    ),
    DemoCase(
        case_id="small_appliances_wrong_or_missing",
        title="Piccoli elettrodomestici errati o incompleti",
        kind="thematic_focus",
        question=(
            "Quali articoli errati o mancanti vengono segnalati nelle recensioni "
            "negative dei piccoli elettrodomestici?"
        ),
        expected_intent="analyze_low_rating_complaints",
        expected_theme="wrong_or_missing_item",
        category="small_appliances",
    ),
    DemoCase(
        case_id="office_furniture_service_refund",
        title="Assistenza o rimborso non risolti per i mobili da ufficio",
        kind="thematic_focus",
        question=(
            "Quali richieste di assistenza o rimborso rimaste senza soluzione "
            "emergono nelle recensioni negative dei mobili per ufficio?"
        ),
        expected_intent="analyze_low_rating_complaints",
        expected_theme="service_or_refund",
        category="office_furniture",
    ),
)

APP_EXAMPLE_CASES: Tuple[DemoCase, ...] = DEMO_CASES + THEMATIC_EXAMPLE_CASES


def get_demo_case(case_id: str) -> DemoCase:
    for case in APP_EXAMPLE_CASES:
        if case.case_id == case_id:
            return case
    raise ValueError("Unknown demo case: {}".format(case_id))


def _check(name: str, passed: bool, observed: Any, expectation: str) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expectation": expectation,
    }


def verify_demo_result(
    case: DemoCase,
    result: Dict[str, Any],
    overall_reference: Dict[str, Any],
) -> Dict[str, Any]:
    checks: List[dict] = []
    question = result["question"]
    evidence = result["context"]["review_evidence"]
    ranked = result["theme_evidence"]["ranked_hypotheses"][:3]
    ranked_names = [item["theme"] for item in ranked]
    checks.append(
        _check(
            "intent",
            question["intent"] == case.expected_intent,
            question["intent"],
            case.expected_intent,
        )
    )
    checks.append(
        _check(
            "requested_theme",
            question.get("requested_theme") == case.expected_theme,
            question.get("requested_theme"),
            str(case.expected_theme),
        )
    )
    checks.append(
        _check(
            "filters",
            question["category"] == case.category
            and question["start_month"] == case.start_month
            and question["end_month"] == case.end_month,
            {
                "category": question["category"],
                "start_month": question["start_month"],
                "end_month": question["end_month"],
            },
            "match the data-driven demo specification",
        )
    )

    if case.kind == "insufficient_evidence":
        checks.extend(
            [
                _check(
                    "insufficient_evidence",
                    result["insufficient_evidence"]["is_insufficient"],
                    result["insufficient_evidence"],
                    "explicit insufficient evidence",
                ),
                _check(
                    "no_qualitative_generation",
                    result["generation"] is None and not evidence,
                    {"generation": result["generation"], "evidence_count": len(evidence)},
                    "no retrieval examples and no LLM generation",
                ),
            ]
        )
    else:
        checks.extend(
            [
                _check(
                    "sufficient_evidence",
                    not result["insufficient_evidence"]["is_insufficient"],
                    result["insufficient_evidence"],
                    "sufficient evidence",
                ),
                _check(
                    "validated_generation",
                    result["generation"] is not None
                    and result["generation"].get("generation_status")
                    in {"llm_generated_validated", "validated_fallback"},
                    (
                        result["generation"].get("generation_status")
                        if result["generation"]
                        else None
                    ),
                    "llm_generated_validated or validated_fallback",
                ),
                _check(
                    "retrieval_evidence",
                    len(evidence) >= (2 if case.kind == "thematic_focus" else 3)
                    and all(
                        item["metadata"].get(
                            "theme_{}".format(item["retrieved_for_theme"])
                        )
                        for item in evidence
                    ),
                    len(evidence),
                    (
                        "at least two reviews for the requested theme"
                        if case.kind == "thematic_focus"
                        else "at least three reviews matching their retrieval theme"
                    ),
                ),
                _check(
                    "translations",
                    all(
                        item.get("translation")
                        and item["translation"].get("status") == "translated"
                        for item in evidence
                    ),
                    [
                        (item.get("translation") or {}).get("status")
                        for item in evidence
                    ],
                    "every retrieved original has an Italian automatic translation",
                ),
            ]
        )

    metrics = result["analytics"]["metrics"]
    comparison = None
    if case.kind == "logistics":
        expected_themes = {"non_delivery", "delivery_delay"}
        checks.extend(
            [
                _check(
                    "rating_drop",
                    metrics["rating_variation"]["value"] < 0,
                    metrics["rating_variation"]["value"],
                    "rating variation below zero",
                ),
                _check(
                    "late_delivery_increase",
                    metrics["late_delivery_rate"]["change"] > 0,
                    metrics["late_delivery_rate"]["change"],
                    "late-delivery rate change above zero",
                ),
                _check(
                    "logistics_themes",
                    expected_themes.issubset(ranked_names),
                    ranked_names,
                    "non_delivery and delivery_delay among the first three hypotheses",
                ),
            ]
        )
    elif case.kind == "product_issues":
        current_rating = metrics["average_rating"]["value"]
        current_late = metrics["late_delivery_rate"]["value"]
        overall_rating = overall_reference["metrics"]["average_rating"]["value"]
        overall_late = overall_reference["metrics"]["late_delivery_rate"]["value"]
        comparison = {
            "rating_difference_vs_overall": current_rating - overall_rating,
            "late_delivery_rate_difference_vs_overall": current_late - overall_late,
            "approximately_stable_delivery_threshold": 0.02,
        }
        checks.extend(
            [
                _check(
                    "rating_below_overall",
                    comparison["rating_difference_vs_overall"] < 0,
                    comparison["rating_difference_vs_overall"],
                    "category rating below overall rating",
                ),
                _check(
                    "delivery_approximately_stable",
                    abs(comparison["late_delivery_rate_difference_vs_overall"]) <= 0.02,
                    comparison["late_delivery_rate_difference_vs_overall"],
                    "absolute difference from overall late-delivery rate <= 0.02",
                ),
                _check(
                    "product_issue_theme",
                    "wrong_or_missing_item" in ranked_names,
                    ranked_names,
                    "wrong_or_missing_item among the first three hypotheses",
                ),
            ]
        )
    elif case.kind == "review_analysis":
        checks.append(
            _check(
                "ranked_complaints",
                len(ranked_names) == 3,
                ranked_names,
                "three distinct ranked complaint hypotheses",
            )
        )
    elif case.kind == "thematic_focus":
        checks.extend(
            [
                _check(
                    "focused_theme_ranking",
                    ranked_names == [case.expected_theme],
                    ranked_names,
                    "only the explicitly requested theme",
                ),
                _check(
                    "focused_theme_retrieval",
                    bool(evidence)
                    and all(
                        item["retrieved_for_theme"] == case.expected_theme
                        for item in evidence
                    ),
                    [item["retrieved_for_theme"] for item in evidence],
                    "all examples match the explicitly requested theme",
                ),
            ]
        )

    return {
        "case": case.as_dict(),
        "comparison": comparison,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
