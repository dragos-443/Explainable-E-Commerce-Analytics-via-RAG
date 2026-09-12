"""Consistency checks for frozen Phase 7 benchmark assets."""

from __future__ import annotations

import json
from pathlib import Path


DATA_ROOT = Path("/workspace/src/ecommerce_rag/evaluation/data")


def _load(name: str):
    return json.loads((DATA_ROOT / name).read_text(encoding="utf-8"))


def test_retrieval_qrels_match_frozen_pool() -> None:
    pool = _load("retrieval_benchmark_pool.json")
    qrels = _load("retrieval_qrels.json")["relevant_ids"]
    assert pool["query_count"] == 20
    for run in pool["runs"]:
        query_id = run["query"]["query_id"]
        pooled_ids = set().union(*(set(ids) for ids in run["methods"].values()))
        assert set(qrels[query_id]).issubset(pooled_ids)


def test_theme_annotations_match_frozen_sample_without_review_text() -> None:
    sample = _load("theme_benchmark_sample.json")
    annotations = _load("theme_annotations.json")["manual_themes"]
    assert len(sample) == 48
    assert {item["review_id"] for item in sample} == set(annotations)
    assert all("review_text_original" not in item for item in sample)


def test_development_predictions_cover_every_annotated_review() -> None:
    annotations = set(_load("theme_annotations.json")["manual_themes"])
    predictions = _load("theme_development_predictions.json")
    assert predictions["baseline"]["classifier_version"] == "rules-pt-v2"
    assert predictions["proposed"]["classifier_version"] == "rules-pt-v3"
    assert set(predictions["baseline"]["predictions"]) == annotations
    assert set(predictions["proposed"]["predictions"]) == annotations


def test_qualitative_annotations_match_frozen_cases() -> None:
    annotations = _load("qualitative_annotations.json")
    cases = _load("qualitative_benchmark_cases.json")
    assert set(cases) == set(annotations["generation"])
    available_ids = {
        item["review_id"]
        for payload in cases.values()
        for item in payload["context"]["review_evidence"]
    }
    assert set(annotations["translations"]).issubset(available_ids)
