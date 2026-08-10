"""Join frozen FinFact accuracy and unified efficiency measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHODS = {
    "minilm": ("baseline", "minilm"),
    "bge_v2_m3": ("baseline", "bge_v2_m3"),
    "mxbai_large": ("baseline", "mxbai_large"),
    "qwen3_reranker_06b": ("baseline", "qwen3_06b"),
    "c2_multitask": ("c2", None),
}


def accuracy(comparison: dict[str, Any], method: str) -> tuple[float, float | None]:
    kind, key = METHODS[method]
    if kind == "c2":
        summary = comparison["c2_three_seed_summary"]
        return float(summary["group_top1_mean"]), float(
            summary["group_top1_sample_std"]
        )
    return float(comparison["baselines"][key]["group"]["accuracy"]), None


def build_rows(comparison: dict[str, Any], efficiency: dict[str, dict]) -> list[dict]:
    if set(efficiency) != set(METHODS):
        raise ValueError("efficiency methods do not match the frozen method set")
    rows = []
    for method in METHODS:
        score, score_std = accuracy(comparison, method)
        value = efficiency[method]
        rows.append(
            {
                "method": method,
                "finfact_group_top1": score,
                "finfact_group_top1_sample_std": score_std,
                "parameters_million": value["parameter_count"] / 1e6,
                "groups_per_second": value["groups_per_second"],
                "candidates_per_second": value["candidates_per_second"],
                "milliseconds_per_group": value["median_milliseconds_per_group"],
                "peak_cuda_memory_mb": value["peak_cuda_memory_mb"],
                "candidate_batch_size": value["candidate_batch_size"],
                "group_batch_size": value["group_batch_size"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--efficiency-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    efficiency = {
        method: json.loads((args.efficiency_dir / f"{method}.json").read_text(encoding="utf-8"))
        for method in METHODS
    }
    rows = build_rows(comparison, efficiency)
    indexed = {row["method"]: row for row in rows}
    c2 = indexed["c2_multitask"]
    qwen = indexed["qwen3_reranker_06b"]
    bge = indexed["bge_v2_m3"]
    report = {
        "status": "complete",
        "method": "publisher_accuracy_efficiency_summary_v1",
        "accuracy_input": str(args.comparison),
        "efficiency_input": str(args.efficiency_dir),
        "rows": rows,
        "contrasts": {
            "c2_vs_strongest_off_the_shelf_qwen3_06b": {
                "top1_delta_percentage_points": 100
                * (c2["finfact_group_top1"] - qwen["finfact_group_top1"]),
                "throughput_ratio": c2["groups_per_second"]
                / qwen["groups_per_second"],
                "peak_memory_reduction_fraction": 1
                - c2["peak_cuda_memory_mb"] / qwen["peak_cuda_memory_mb"],
            },
            "c2_vs_bge_backbone": {
                "top1_delta_percentage_points": 100
                * (c2["finfact_group_top1"] - bge["finfact_group_top1"]),
                "latency_overhead_fraction": c2["milliseconds_per_group"]
                / bge["milliseconds_per_group"]
                - 1,
                "parameter_overhead_fraction": c2["parameters_million"]
                / bge["parameters_million"]
                - 1,
                "peak_memory_overhead_fraction": c2["peak_cuda_memory_mb"]
                / bge["peak_cuda_memory_mb"]
                - 1,
            },
        },
        "reporting_policy": [
            "Accuracy and efficiency use the same frozen FinFact benchmark contract.",
            "C2 accuracy is the predefined three-seed mean; off-the-shelf models are frozen single checkpoints.",
            "Efficiency is the median of three post-warmup inference-only repetitions.",
            "artifact_bytes is excluded because model directories may contain duplicate serialization formats.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
