"""Paired comparison of two frozen publisher-router variants across seeds.

Both prediction files must be produced by
``export_multitask_publisher_predictions_v1`` on the same frozen input.
Repeated seeds are summarized, but significance is reported separately per seed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from provtimerag.data import read_bundles_jsonl


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows = {
        row["group_id"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    if not rows:
        raise ValueError(f"empty predictions: {path}")
    return rows


def exact_mcnemar(improvements: int, regressions: int) -> float:
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    tail = min(improvements, regressions)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * probability)


def paired_bootstrap(
    first: list[bool],
    second: list[bool],
    *,
    samples: int = 10_000,
    seed: int = 20260810,
) -> list[float]:
    if len(first) != len(second) or not first:
        raise ValueError("paired non-empty observations are required")
    rng = random.Random(seed)
    count = len(first)
    deltas = []
    for _ in range(samples):
        total = 0
        for _ in range(count):
            index = rng.randrange(count)
            total += int(first[index]) - int(second[index])
        deltas.append(total / count)
    deltas.sort()
    return [deltas[int(0.025 * samples)], deltas[int(0.975 * samples)]]


def load_bundles(path: Path) -> dict[str, list[str]]:
    return {
        bundle.bundle_id: [group.group_id for group in bundle.groups]
        for bundle in read_bundles_jsonl(path)
    }


def hits(rows: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {group_id: bool(row["hit"]) for group_id, row in rows.items()}


def bundle_hits(
    group_hits: dict[str, bool], bundles: dict[str, list[str]]
) -> dict[str, bool]:
    expected = {group_id for group_ids in bundles.values() for group_id in group_ids}
    if set(group_hits) != expected:
        raise ValueError("prediction population does not match frozen input")
    return {
        bundle_id: all(group_hits[group_id] for group_id in group_ids)
        for bundle_id, group_ids in bundles.items()
    }


def paired_report(first: dict[str, bool], second: dict[str, bool]) -> dict[str, Any]:
    keys = sorted(first)
    if keys != sorted(second):
        raise ValueError("paired variants do not have identical populations")
    first_values = [first[key] for key in keys]
    second_values = [second[key] for key in keys]
    improvements = sum(a and not b for a, b in zip(first_values, second_values, strict=True))
    regressions = sum(b and not a for a, b in zip(first_values, second_values, strict=True))
    first_accuracy = sum(first_values) / len(keys)
    second_accuracy = sum(second_values) / len(keys)
    return {
        "items": len(keys),
        "first_accuracy": first_accuracy,
        "second_accuracy": second_accuracy,
        "delta": first_accuracy - second_accuracy,
        "improvements": improvements,
        "regressions": regressions,
        "exact_two_sided_mcnemar_p": exact_mcnemar(improvements, regressions),
        "paired_bootstrap_95_ci": paired_bootstrap(first_values, second_values),
    }


def frequency_slices(
    first: dict[str, bool],
    second: dict[str, bool],
    rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[str]] = {}
    for group_id, row in rows.items():
        bucket = str(row.get("publisher_frequency_bucket", "unknown"))
        buckets.setdefault(bucket, []).append(group_id)
    return {
        bucket: paired_report(
            {group_id: first[group_id] for group_id in group_ids},
            {group_id: second[group_id] for group_id in group_ids},
        )
        for bucket, group_ids in sorted(buckets.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--first-name", required=True)
    parser.add_argument("--second-name", required=True)
    parser.add_argument("--first", action="append", type=named_path, required=True)
    parser.add_argument("--second", action="append", type=named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    first_paths = dict(args.first)
    second_paths = dict(args.second)
    if set(first_paths) != set(second_paths):
        raise ValueError("first and second variants must provide identical seed names")
    if len(first_paths) < 2:
        raise ValueError("at least two predefined seeds are required")

    bundles = load_bundles(args.input)
    per_seed: dict[str, Any] = {}
    first_group_scores = []
    second_group_scores = []
    first_bundle_scores = []
    second_bundle_scores = []
    for index, seed in enumerate(sorted(first_paths, key=lambda value: int(value))):
        first_rows = read_predictions(first_paths[seed])
        second_rows = read_predictions(second_paths[seed])
        first_group = hits(first_rows)
        second_group = hits(second_rows)
        first_bundle = bundle_hits(first_group, bundles)
        second_bundle = bundle_hits(second_group, bundles)
        group_report = paired_report(first_group, second_group)
        bundle_report = paired_report(first_bundle, second_bundle)
        first_group_scores.append(group_report["first_accuracy"])
        second_group_scores.append(group_report["second_accuracy"])
        first_bundle_scores.append(bundle_report["first_accuracy"])
        second_bundle_scores.append(bundle_report["second_accuracy"])
        per_seed[seed] = {
            "group": group_report,
            "bundle": bundle_report,
            "publisher_frequency_slices": frequency_slices(
                first_group, second_group, first_rows
            ),
            "bootstrap_seed": 20260810,
        }

    def summary(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }

    report = {
        "status": "complete",
        "method": "frozen_publisher_router_variant_paired_comparison_v1",
        "input": str(args.input),
        "first_name": args.first_name,
        "second_name": args.second_name,
        "seeds": sorted(first_paths, key=lambda value: int(value)),
        "reporting_policy": [
            "Report mean and sample standard deviation over predefined seeds.",
            "Do not pool repeated seed predictions as independent observations.",
            "Report paired significance separately for every seed and population.",
        ],
        "summary": {
            "first_group_top1": summary(first_group_scores),
            "second_group_top1": summary(second_group_scores),
            "group_delta": summary(
                [a - b for a, b in zip(first_group_scores, second_group_scores, strict=True)]
            ),
            "first_bundle_exact": summary(first_bundle_scores),
            "second_bundle_exact": summary(second_bundle_scores),
            "bundle_delta": summary(
                [a - b for a, b in zip(first_bundle_scores, second_bundle_scores, strict=True)]
            ),
        },
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
