"""Score manually reviewed generation and translation samples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DATA_ROOT = Path(__file__).with_name("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", default="/workspace/reports/demo/phase6")
    parser.add_argument("--output-root", default="/workspace/reports/evaluation/core/phase7")
    parser.add_argument(
        "--input-source",
        choices=("reference", "latest"),
        default="reference",
        help="Use the frozen manually reviewed cases or the latest demo outputs.",
    )
    return parser.parse_args()


def average(items, key: str) -> float:
    return sum(float(item[key]) for item in items) / len(items)


def main() -> None:
    args = parse_args()
    annotations = json.loads(
        (DATA_ROOT / "qualitative_annotations.json").read_text(encoding="utf-8")
    )
    if args.input_source == "reference":
        payloads = json.loads(
            (DATA_ROOT / "qualitative_benchmark_cases.json").read_text(encoding="utf-8")
        )
    else:
        demo_root = Path(args.demo_root)
        payloads = {
            case_id: json.loads(
                (demo_root / "{}.json".format(case_id)).read_text(encoding="utf-8")
            )
            for case_id in annotations["generation"]
        }
    generation_rows = []
    available_translations = {}
    for case_id, payload in payloads.items():
        generation = payload["generation"]
        evidence = payload["context"]["review_evidence"]
        available_ids = {item["review_id"] for item in evidence}
        available_themes = {
            item["theme"] for item in payload["context"]["ranked_theme_evidence"]
        }
        for item in evidence:
            available_translations[item["review_id"]] = item
        if generation is None:
            automated_grounding = (
                payload["insufficient_evidence"]["is_insufficient"] and not evidence
            )
        else:
            automated_grounding = (
                set(generation["review_ids"]).issubset(available_ids)
                and set(generation["theme_keys"]).issubset(available_themes)
                and not re.search(r"\d", generation["interpretation"])
                and generation["generation_status"] == "llm_generated_validated"
            )
        generation_rows.append(
            {
                "case_id": case_id,
                **annotations["generation"][case_id],
                "automated_grounding_passed": bool(automated_grounding),
            }
        )
    translation_rows = []
    for review_id, judgment in annotations["translations"].items():
        if review_id not in available_translations:
            raise ValueError("Translation annotation references unknown review {}".format(review_id))
        item = available_translations[review_id]
        translation_rows.append(
            {
                "review_id": review_id,
                **judgment,
                "translation_status": item["translation"]["status"],
            }
        )
    applicable = {
        key: [row[key] for row in translation_rows if row[key] is not None]
        for key in (
            "negation_preserved",
            "dates_quantities_preserved",
            "complaint_subject_preserved",
            "overall_meaning_preserved",
        )
    }
    threshold = annotations["annotation_protocol"]["translation_acceptability_threshold"]
    result = {
        "schema_version": "1.0",
        "annotation_protocol": annotations["annotation_protocol"],
        "input_source": args.input_source,
        "generation": {
            "sample_size": len(generation_rows),
            "mean_scores": {
                key: average(generation_rows, key)
                for key in ("groundedness", "correctness", "answer_relevance", "clarity")
            },
            "automated_grounding_pass_rate": sum(row["automated_grounding_passed"] for row in generation_rows) / len(generation_rows),
            "cases": generation_rows,
        },
        "translation": {
            "sample_size": len(translation_rows),
            "mean_fidelity": average(translation_rows, "fidelity"),
            "acceptable_rate": sum(row["fidelity"] >= threshold for row in translation_rows) / len(translation_rows),
            "preservation_rates": {
                key: sum(values) / len(values) for key, values in applicable.items()
            },
            "cases": translation_rows,
        },
        "capability_ablation": {
            "analytics_only": {"kpi_metrics": True, "population_prevalence": False, "review_examples": False, "validated_synthesis": False},
            "reviews_only": {"kpi_metrics": False, "population_prevalence": False, "review_examples": True, "validated_synthesis": False},
            "analytics_plus_rag": {"kpi_metrics": True, "population_prevalence": True, "review_examples": True, "validated_synthesis": True}
        }
    }
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "qualitative_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PHASE7_QUALITATIVE="
        + json.dumps(
            {
                "generation_sample": len(generation_rows),
                "grounding_pass_rate": result["generation"]["automated_grounding_pass_rate"],
                "translation_sample": len(translation_rows),
                "translation_mean_fidelity": result["translation"]["mean_fidelity"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
