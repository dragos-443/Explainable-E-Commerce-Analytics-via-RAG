"""Pure evaluation metrics used by the Phase 7 experiments."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Set


def retrieval_metrics(
    ranked_ids: Sequence[str], relevant_ids: Set[str], k: int
) -> Dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")
    selected = list(ranked_ids[:k])
    hits = sum(document_id in relevant_ids for document_id in selected)
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant_ids) if relevant_ids else 0.0,
        "hit_rate_at_k": float(hits > 0),
        "hits_at_k": hits,
    }


def macro_average(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    values = list(rows)
    if not values:
        raise ValueError("at least one metric row is required")
    keys = values[0].keys()
    return {key: sum(float(row[key]) for row in values) / len(values) for key in keys}


def multilabel_metrics(
    expected_by_id: Mapping[str, Set[str]], predicted_by_id: Mapping[str, Set[str]]
) -> Dict[str, object]:
    if set(expected_by_id) != set(predicted_by_id):
        raise ValueError("expected and predicted ids must match")
    labels = sorted(
        set().union(*expected_by_id.values(), *predicted_by_id.values())
        if expected_by_id
        else set()
    )
    per_label = {}
    total_tp = total_fp = total_fn = 0
    for label in labels:
        tp = sum(label in expected_by_id[key] and label in predicted_by_id[key] for key in expected_by_id)
        fp = sum(label not in expected_by_id[key] and label in predicted_by_id[key] for key in expected_by_id)
        fn = sum(label in expected_by_id[key] and label not in predicted_by_id[key] for key in expected_by_id)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "per_label": per_label,
        "macro_precision": sum(item["precision"] for item in per_label.values()) / len(per_label) if per_label else 0.0,
        "macro_recall": sum(item["recall"] for item in per_label.values()) / len(per_label) if per_label else 0.0,
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(per_label) if per_label else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "exact_match_ratio": sum(expected_by_id[key] == predicted_by_id[key] for key in expected_by_id) / len(expected_by_id) if expected_by_id else 0.0,
    }
