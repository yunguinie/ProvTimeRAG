"""Summarize corrected frozen external C3 metrics across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate.summarize_c3_frozen_external_v1 import summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    policy_ids = {row["frozen_policy_id"] for row in rows}
    holdout_hashes = {row["input_sha256"]["holdout"] for row in rows}
    policy_hashes = {row["input_sha256"]["frozen_policy"] for row in rows}
    if len(policy_ids) != 1 or len(holdout_hashes) != 1 or len(policy_hashes) != 1:
        raise ValueError("seeds do not share one frozen policy and holdout")
    pairs = {
        "group_top1": ("independent", "top1_accuracy"),
        "independent_bundle_exact": ("independent", "bundle_exact_match"),
        "frozen_robust_bundle_exact": ("frozen_robust", "bundle_exact_match"),
        "independent_source_set_exact": ("independent", "source_set_exact_match"),
        "frozen_robust_source_set_exact": ("frozen_robust", "source_set_exact_match"),
        "independent_source_set_jaccard": ("independent", "source_set_jaccard"),
        "frozen_robust_source_set_jaccard": ("frozen_robust", "source_set_jaccard"),
        "independent_source_count_mismatch": ("independent", "source_count_mismatch_rate"),
        "frozen_robust_source_count_mismatch": ("frozen_robust", "source_count_mismatch_rate"),
        "independent_binary_multi_source_error": ("independent", "binary_multi_source_error_rate"),
        "frozen_robust_binary_multi_source_error": ("frozen_robust", "binary_multi_source_error_rate"),
        "correction_coverage": ("frozen_robust", "correction_coverage"),
    }
    metrics = {
        name: summary([row["metrics"][method][field] for row in rows])
        for name, (method, field) in pairs.items()
    }
    metrics["bundle_exact_delta"] = summary(
        [row["paired_significance"]["bundle_exact_delta"] for row in rows]
    )
    metrics["source_set_exact_delta"] = summary(
        [
            row["metrics"]["frozen_robust"]["source_set_exact_match"]
            - row["metrics"]["independent"]["source_set_exact_match"]
            for row in rows
        ]
    )
    report = {
        "status": "complete",
        "method": "frozen_robust_c3_external_v3_visible_ties_complete_source_metrics",
        "seeds": [row["seed"] for row in rows],
        "frozen_policy_id": next(iter(policy_ids)),
        "holdout_sha256": next(iter(holdout_hashes)),
        "frozen_policy_sha256": next(iter(policy_hashes)),
        "metrics": metrics,
        "paired_significance_by_seed": [
            {"seed": row["seed"], **row["paired_significance"]} for row in rows
        ],
        "metric_notes": [
            "source_set_exact_match compares the complete predicted and gold publisher sets.",
            "binary_multi_source_error only checks the single-source versus multi-source class.",
            "The legacy total_source_error name is intentionally not used.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
