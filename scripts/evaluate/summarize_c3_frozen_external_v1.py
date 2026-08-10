"""Summarize frozen external C3 results across independently trained seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policy_ids = {row["frozen_policy_id"] for row in rows}
    holdout_hashes = {row["input_sha256"]["holdout"] for row in rows}
    policy_hashes = {row["input_sha256"]["frozen_policy"] for row in rows}
    if len(policy_ids) != 1 or len(holdout_hashes) != 1 or len(policy_hashes) != 1:
        raise ValueError("seeds do not share one frozen policy and holdout")
    metrics = {
        "group_top1": [row["metrics"]["independent"]["top1_accuracy"] for row in rows],
        "independent_bundle_exact": [
            row["metrics"]["independent"]["bundle_exact_match"] for row in rows
        ],
        "frozen_robust_bundle_exact": [
            row["metrics"]["frozen_robust"]["bundle_exact_match"] for row in rows
        ],
        "bundle_exact_delta": [
            row["paired_significance"]["bundle_exact_delta"] for row in rows
        ],
        "independent_total_source_error": [
            row["metrics"]["independent"]["total_source_error"] for row in rows
        ],
        "frozen_robust_total_source_error": [
            row["metrics"]["frozen_robust"]["total_source_error"] for row in rows
        ],
        "correction_coverage": [
            row["metrics"]["frozen_robust"]["correction_coverage"] for row in rows
        ],
    }
    return {
        "status": "complete",
        "method": "frozen_cardinality_shift_robust_c3_external_v1_three_seed_summary",
        "seeds": [row["seed"] for row in rows],
        "frozen_policy_id": next(iter(policy_ids)),
        "holdout_sha256": next(iter(holdout_hashes)),
        "frozen_policy_sha256": next(iter(policy_hashes)),
        "metrics": {name: summary(values) for name, values in metrics.items()},
        "paired_significance_by_seed": [
            {"seed": row["seed"], **row["paired_significance"]} for row in rows
        ],
        "interpretation_policy": [
            "Report mean and sample standard deviation over the three predefined seeds.",
            "Do not pool repeated predictions across seeds as independent observations.",
            "Report paired significance separately for each seed on the common frozen holdout.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    report = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
