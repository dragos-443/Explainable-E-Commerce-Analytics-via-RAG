"""Build structured Phase 7 summaries and plots from measured artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/workspace/reports/evaluation/phase7")
    return parser.parse_args()


def load(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def retrieval_plot(root: Path, payload: dict) -> None:
    labels = ["Precision@5", "Pooled Recall@5", "Hit Rate@5"]
    keys = ["precision_at_k", "recall_at_k", "hit_rate_at_k"]
    baseline = payload["macro_metrics"]["semantic_only"]
    filtered = payload["macro_metrics"]["metadata_plus_semantic"]
    reranked = payload["macro_metrics"]["metadata_plus_semantic_reranked"]
    positions = list(range(len(labels)))
    width = 0.25
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar([x - width for x in positions], [baseline[key] for key in keys], width, label="Solo semantico")
    axis.bar(positions, [filtered[key] for key in keys], width, label="Filtri + semantico")
    axis.bar([x + width for x in positions], [reranked[key] for key in keys], width, label="Top-50 + reranker")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Valore medio")
    axis.set_title("Retrieval cross-lingue italiano-portoghese")
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "retrieval_comparison.png", dpi=160)
    plt.close(fig)


def theme_plot(root: Path, payload: dict) -> None:
    metrics = payload["final_metrics"]["per_label"]
    labels = list(metrics)
    values = [metrics[label]["f1"] for label in labels]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.barh(labels, values, color="#4472C4")
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("F1")
    axis.set_title("F1 per complaint theme sul campione manuale")
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(root / "theme_f1.png", dpi=160)
    plt.close(fig)


def theme_improvement_plot(root: Path, payload: dict) -> None:
    labels = ["Micro-F1", "Macro-F1", "Exact match"]
    keys = ["micro_f1", "macro_f1", "exact_match_ratio"]
    baseline = payload["baseline"]["metrics"]
    proposed = payload["proposed"]["metrics"]
    positions = list(range(len(labels)))
    width = 0.35
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        [x - width / 2 for x in positions],
        [baseline[key] for key in keys],
        width,
        label="rules-pt-v2",
    )
    axis.bar(
        [x + width / 2 for x in positions],
        [proposed[key] for key in keys],
        width,
        label="rules-pt-v3",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Valore")
    axis.set_title("Classificatori dei temi sul nuovo holdout")
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "theme_classifier_improvement.png", dpi=160)
    plt.close(fig)


def latency_plot(root: Path, payload: dict) -> None:
    keys = [
        "analytics_spark_total",
        "query_embedding",
        "chroma_retrieval",
        "reranking",
        "translation_top_k",
        "llm_generation",
        "pipeline_other",
    ]
    labels = ["Spark", "Query embedding", "Chroma", "Reranking", "Traduzione", "LLM", "Altro"]
    cold = payload["online"]["cold"]
    warm = payload["online"]["warm_summary"]
    cold_values = [cold[key] for key in keys]
    warm_values = [warm[key]["mean_seconds"] for key in keys]
    fig, axis = plt.subplots(figsize=(9, 5))
    bottoms = [0.0, 0.0]
    for label, cold_value, warm_value in zip(labels, cold_values, warm_values):
        axis.bar(["Fredda", "Calda (media)"], [cold_value, warm_value], bottom=bottoms, label=label)
        bottoms = [bottoms[0] + cold_value, bottoms[1] + warm_value]
    axis.set_ylabel("Secondi")
    axis.set_title("Scomposizione della latenza end-to-end")
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(root / "latency_breakdown.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    retrieval = load(root, "retrieval_metrics.json")
    themes = load(root, "theme_classification_metrics.json")
    theme_improvement = load(root, "theme_classifier_improvement_metrics.json")
    qualitative = load(root, "qualitative_metrics.json")
    efficiency = load(root, "efficiency_metrics.json")
    retrieval_plot(root, retrieval)
    theme_plot(root, themes)
    theme_improvement_plot(root, theme_improvement)
    latency_plot(root, efficiency)
    false_positive = Counter()
    false_negative = Counter()
    for error in themes["errors"]:
        false_positive.update(error["false_positive"])
        false_negative.update(error["false_negative"])
    zero_relevant_queries = sorted(
        {
            row["query_id"]
            for row in retrieval["per_query"]
            if row["relevant_in_pool"] == 0
        }
    )
    warm = efficiency["online"]["warm_summary"]
    component_means = {
        key: warm[key]["mean_seconds"]
        for key in (
            "analytics_spark_total", "query_embedding", "chroma_retrieval",
            "reranking", "translation_top_k", "llm_generation", "pipeline_other",
        )
    }
    dominant = max(component_means, key=component_means.get)
    summary = {
        "schema_version": "1.0",
        "retrieval": {
            "query_count": retrieval["query_count"],
            "judged_pairs": sum(
                len(set().union(*[set(ids) for ids in run["methods"].values()]))
                for run in load(root, "retrieval_pool.json")["runs"]
            ),
            "semantic_only": retrieval["macro_metrics"]["semantic_only"],
            "metadata_plus_semantic": retrieval["macro_metrics"]["metadata_plus_semantic"],
            "metadata_plus_semantic_reranked": retrieval["macro_metrics"]["metadata_plus_semantic_reranked"],
            "candidate_k": retrieval["candidate_k"],
            "reranker": retrieval["reranker"],
            "zero_relevant_pool_queries": zero_relevant_queries,
        },
        "theme_classification": {
            "pilot_size": themes["composition"]["pilot_size"],
            "final_size": themes["composition"]["sample_size"],
            "micro_f1": themes["final_metrics"]["micro_f1"],
            "macro_f1": themes["final_metrics"]["macro_f1"],
            "exact_match_ratio": themes["final_metrics"]["exact_match_ratio"],
            "most_common_false_positives": false_positive.most_common(),
            "most_common_false_negatives": false_negative.most_common(),
        },
        "theme_classifier_improvement": {
            "holdout_size": theme_improvement["sample_size"],
            "baseline_version": theme_improvement["baseline"]["classifier_version"],
            "baseline_metrics": theme_improvement["baseline"]["metrics"],
            "proposed_version": theme_improvement["proposed"]["classifier_version"],
            "proposed_metrics": theme_improvement["proposed"]["metrics"],
        },
        "generation": qualitative["generation"]["mean_scores"],
        "translation": {
            "sample_size": qualitative["translation"]["sample_size"],
            "mean_fidelity": qualitative["translation"]["mean_fidelity"],
            "acceptable_rate": qualitative["translation"]["acceptable_rate"],
        },
        "efficiency": {
            "full_index_seconds": efficiency["offline"]["full_initial_build"]["total_seconds"],
            "full_index_documents": efficiency["offline"]["full_initial_build"]["documents"],
            "chroma_index_bytes": efficiency["storage"]["chroma_index_bytes"],
            "translation_cache_bytes": efficiency["storage"]["translation_cache_bytes_before_benchmark"],
            "cold_end_to_end_seconds": efficiency["online"]["cold"]["end_to_end"],
            "warm_end_to_end_mean_seconds": warm["end_to_end"]["mean_seconds"],
            "warm_component_means_seconds": component_means,
            "dominant_warm_component": dominant,
            "dominant_warm_share": component_means[dominant] / warm["end_to_end"]["mean_seconds"],
        },
        "plots": [
            "retrieval_comparison.png",
            "theme_f1.png",
            "theme_classifier_improvement.png",
            "latency_breakdown.png",
        ],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE7_SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
