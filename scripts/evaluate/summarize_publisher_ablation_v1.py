"""Summarize frozen three-seed publisher ablations and training cost."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


FIELDS = (
    "training_groups",
    "optimizer_steps",
    "training_seconds",
    "dev_publisher_top1",
    "dev_temporal_top1",
    "dev_insufficient_f1",
    "clean_blind_top1",
    "finfact_top1",
    "finfact_false_abstention_rate",
)


def named_template(value: str) -> tuple[str, str]:
    name, separator, template = value.partition("=")
    if not separator or "{seed}" not in template:
        raise argparse.ArgumentTypeError("expected NAME=PATH_WITH_{seed}")
    return name, template


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("status") != "complete":
        raise ValueError(f"incomplete result: {path}")
    return row


def exposed_training_groups(train: dict[str, Any], history: dict[str, Any]) -> int:
    if "training_groups" in history:
        return int(history["training_groups"])
    if "groups_per_task" in train:
        return 3 * int(train["groups_per_task"])
    raise KeyError("training_groups or groups_per_task")


def seed_row(name: str, seed: int, root: Path) -> tuple[dict, dict[str, str]]:
    train = load(root / "metrics.json")
    blind = load(root / "c3_blind_v3_clean_metrics_fp32.json")
    finfact = load(root / "finfact_metrics_fp32.json")
    history = train["history"][-1]
    dev = train["dev"]
    finfact_abstention = finfact["metrics"]["abstention"]
    row = {
        "variant": name,
        "seed": seed,
        "source_url_included": bool(train["source_url_included"]),
        "training_groups": exposed_training_groups(train, history),
        "optimizer_steps": int(history["optimizer_steps"]),
        "training_seconds": float(train["elapsed_seconds"]),
        "dev_publisher_top1": float(dev["source"]["top1_accuracy"]),
        "dev_temporal_top1": float(dev["temporal_version"]["top1_accuracy"]),
        "dev_insufficient_f1": float(dev["insufficient"]["abstention"]["f1"]),
        "clean_blind_top1": float(blind["metrics"]["top1_accuracy"]),
        "finfact_top1": float(finfact["metrics"]["top1_accuracy"]),
        "finfact_false_abstention_rate": float(
            finfact_abstention["fp"] / finfact_abstention["support"]
        ),
    }
    hashes = {
        "source_train": str(blind["input_sha256"]["source_train"]),
        "clean_blind": str(blind["input_sha256"]["holdout"]),
        "finfact": str(finfact["input_sha256"]["holdout"]),
    }
    return row, hashes


def summarize(rows: list[dict]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in FIELDS:
        values = [float(row[field]) for row in rows]
        result[field] = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }
    result["source_url_included"] = rows[0]["source_url_included"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", type=named_template, required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    variants = dict(args.variant)
    seeds = sorted(set(args.seed))
    if len(variants) < 2 or len(seeds) < 3:
        raise ValueError("at least two variants and three predefined seeds are required")
    rows: list[dict] = []
    hashes: dict[str, str] | None = None
    for name, template in variants.items():
        for seed in seeds:
            row, current_hashes = seed_row(name, seed, Path(template.format(seed=seed)))
            if hashes is None:
                hashes = current_hashes
            elif hashes != current_hashes:
                raise ValueError("variants do not use identical frozen input hashes")
            rows.append(row)
    summaries = {
        name: summarize([row for row in rows if row["variant"] == name])
        for name in variants
    }
    reference_name = next(iter(variants))
    reference = summaries[reference_name]
    contrasts = {
        name: {
            field: summary[field]["mean"] - reference[field]["mean"]
            for field in FIELDS
        }
        for name, summary in summaries.items()
        if name != reference_name
    }
    report = {
        "status": "complete",
        "method": "frozen_publisher_ablation_three_seed_summary_v1",
        "reference_variant": reference_name,
        "seeds": seeds,
        "input_sha256": hashes,
        "per_seed": rows,
        "summary": summaries,
        "mean_delta_vs_reference": contrasts,
        "reporting_policy": [
            "Report mean and sample standard deviation across predefined seeds.",
            "Data-matched source-only uses one pass over the unique source groups; legacy repeated compute-matched source-only is excluded.",
            "Optimizer steps and exposed training groups are the primary training-cost measures; wall-clock seconds are supplementary.",
            "All external metrics must share identical frozen source-train and holdout hashes.",
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
