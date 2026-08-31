"""Integration checks for measured Phase 7 evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/workspace/reports/evaluation/phase7")
DATA_ROOT = Path("/workspace/src/ecommerce_rag/evaluation/data")


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_cross_language_retrieval_compares_two_methods_on_judged_pool() -> None:
    payload = _load("retrieval_metrics.json")
    baseline = payload["macro_metrics"]["semantic_only"]
    filtered = payload["macro_metrics"]["metadata_plus_semantic"]
    proposed = payload["macro_metrics"]["metadata_plus_semantic_reranked"]
    assert payload["query_count"] == 20
    assert payload["query_language"] == "it"
    assert payload["evidence_language"] == "pt"
    assert "judged pool" in payload["recall_definition"]
    assert payload["candidate_k"] == 50
    for metric in ("precision_at_k", "recall_at_k", "hit_rate_at_k"):
        assert proposed[metric] > baseline[metric]
    assert proposed["precision_at_k"] > filtered["precision_at_k"]


def test_retrieval_evaluation_uses_real_italian_questions() -> None:
    questions = json.loads(
        (DATA_ROOT / "retrieval_queries.json").read_text(encoding="utf-8")
    )
    assert len(questions) == 20
    assert all(item["question_it"].strip().endswith("?") for item in questions)


def test_manual_theme_sample_records_pilot_final_metrics_and_errors() -> None:
    payload = _load("theme_classification_metrics.json")
    assert payload["composition"]["pilot_size"] == 25
    assert payload["composition"]["sample_size"] == 48
    assert payload["composition"]["sampling_per_predicted_theme"] == 6
    assert payload["composition"]["categories"] >= 10
    assert payload["composition"]["months"] == 12
    assert 0 <= payload["final_metrics"]["micro_f1"] <= 1
    assert 0 <= payload["final_metrics"]["macro_f1"] <= 1
    assert payload["errors"]
    assert {"other", "uncertain"}.issubset(payload["final_metrics"]["per_label"])


def test_improved_theme_classifier_uses_a_separate_blind_holdout() -> None:
    payload = _load("theme_classifier_improvement_metrics.json")
    baseline = payload["baseline"]["metrics"]
    proposed = payload["proposed"]["metrics"]
    assert payload["sample_size"] == 48
    assert payload["annotation_protocol"]["automatic_predictions_visible_during_annotation"] is False
    assert payload["baseline"]["classifier_version"] == "rules-pt-v2"
    assert payload["proposed"]["classifier_version"] == "rules-pt-v3"
    assert proposed["micro_precision"] > baseline["micro_precision"]
    assert proposed["macro_f1"] > baseline["macro_f1"]
    assert proposed["exact_match_ratio"] > baseline["exact_match_ratio"]


def test_generation_and_translation_are_evaluated_separately() -> None:
    payload = _load("qualitative_metrics.json")
    assert payload["generation"]["sample_size"] == 4
    assert payload["generation"]["automated_grounding_pass_rate"] == 1.0
    assert payload["translation"]["sample_size"] == 12
    assert payload["translation"]["mean_fidelity"] == 4.0
    assert payload["translation"]["acceptable_rate"] == 1.0
    assert payload["capability_ablation"]["analytics_plus_rag"] == {
        "kpi_metrics": True,
        "population_prevalence": True,
        "review_examples": True,
        "validated_synthesis": True,
    }


def test_efficiency_separates_offline_cold_and_warm_measurements() -> None:
    payload = _load("efficiency_metrics.json")
    assert payload["offline"]["full_initial_build"]["documents"] == 42_380
    assert payload["offline"]["full_initial_build"]["total_seconds"] > 0
    assert payload["offline"]["controlled_sample"]["documents"] == 512
    assert payload["storage"]["chroma_index_bytes"] > 0
    assert payload["storage"]["translation_cache_bytes_before_benchmark"] > 0
    repetitions = payload["online"]["repetitions"]
    assert len(repetitions) == 3
    assert repetitions[0]["translation_cache_misses"] == 3
    assert all(row["translation_cache_hits"] == 3 for row in repetitions[1:])
    assert payload["online"]["warm_summary"]["end_to_end"]["mean_seconds"] > 0
    assert payload["configuration"]["retrieval_candidate_k"] == 50
    assert payload["online"]["warm_summary"]["reranking"]["mean_seconds"] > 0


def test_summary_and_plots_exist() -> None:
    summary = _load("summary.json")
    assert summary["retrieval"]["judged_pairs"] == 242
    assert summary["efficiency"]["dominant_warm_component"] == "llm_generation"
    for name in summary["plots"]:
        path = ROOT / name
        assert path.exists()
        assert path.stat().st_size > 10_000
