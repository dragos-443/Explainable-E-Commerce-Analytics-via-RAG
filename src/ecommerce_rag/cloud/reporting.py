"""Build local-versus-EMR comparison artifacts for Phase 10."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def comparison_rows(
    local_rows: list[dict[str, Any]], cloud_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    local = {(row["factor"], row["workload"]): row for row in local_rows}
    cloud = {(row["factor"], row["workload"]): row for row in cloud_rows}
    common = sorted(local.keys() & cloud.keys())
    rows = []
    for factor, workload in common:
        local_row = local[(factor, workload)]
        cloud_row = cloud[(factor, workload)]
        local_time = float(local_row["mean_execution_seconds"])
        cloud_time = float(cloud_row["mean_execution_seconds"])
        rows.append(
            {
                "factor": factor,
                "workload": workload,
                "local_mean_seconds": local_time,
                "cloud_mean_seconds": cloud_time,
                "cloud_speedup_vs_local": local_time / cloud_time,
                "local_throughput_rows_per_second": float(
                    local_row["mean_throughput_rows_per_second"]
                ),
                "cloud_throughput_rows_per_second": float(
                    cloud_row["mean_throughput_rows_per_second"]
                ),
            }
        )
    return rows


def build_plots(rows: list[dict[str, Any]], output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "A_simple_aggregation": "A - aggregazione",
        "B_join_aggregation": "B - join + aggregazione",
        "C_multi_join_features_aggregation": "C - multi-join + feature",
    }
    workloads = sorted({row["workload"] for row in rows})
    fig, axes = plt.subplots(1, len(workloads), figsize=(15, 4.5), squeeze=False)
    for axis, workload in zip(axes[0], workloads):
        selected = sorted(
            (row for row in rows if row["workload"] == workload),
            key=lambda row: row["factor"],
        )
        factors = [row["factor"] for row in selected]
        axis.plot(
            factors,
            [row["local_mean_seconds"] for row in selected],
            marker="o",
            label="Locale",
        )
        axis.plot(
            factors,
            [row["cloud_mean_seconds"] for row in selected],
            marker="o",
            label="AWS EMR",
        )
        axis.set_title(labels.get(workload, workload))
        axis.set_xlabel("Fattore")
        axis.set_ylabel("Tempo medio (s)")
        axis.set_xticks(factors)
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_root / "execution_time_local_vs_aws.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True)
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    local = json.loads(Path(args.local).read_text(encoding="utf-8"))
    cloud = json.loads(Path(args.cloud).read_text(encoding="utf-8"))
    rows = comparison_rows(local["summary"], cloud["summary"])
    if not rows:
        raise ValueError("local and cloud summaries have no common configurations")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    (output_root / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    from ecommerce_rag.scalability.phase9 import build_plots as build_scalability_plots

    build_scalability_plots(cloud["summary"], Path(args.cloud).parent)
    build_plots(rows, output_root)
    print("PHASE10_COMPARISON=" + json.dumps({"rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
