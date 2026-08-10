"""Evaluate one previously frozen robust C3 policy on an untouched holdout."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from scripts.evaluate import run_c3_external_baselines_v1 as external
from scripts.evaluate.run_c3_cardinality_robust_dev_v1 import report_policy, sha256_file


def selected_configuration(path: Path) -> tuple[str, dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = report.get("selected") or {}
    configuration = selected.get("configuration") or {}
    required = {"alpha", "max_regret_ratio", "max_source_count_delta"}
    if set(configuration) < required:
        raise ValueError(f"selected policy is missing {sorted(required - set(configuration))}")
    return str(report.get("selected_policy") or "unknown"), configuration


def exact_two_sided_binomial(improvements: int, regressions: int) -> float:
    discordant = improvements + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(improvements, regressions) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_bootstrap_ci(
    baseline_hits: list[int], method_hits: list[int], seed: int, samples: int = 10000
) -> list[float]:
    rng = random.Random(seed)
    differences = []
    count = len(baseline_hits)
    for _ in range(samples):
        delta = 0
        for _ in range(count):
            index = rng.randrange(count)
            delta += method_hits[index] - baseline_hits[index]
        differences.append(delta / count)
    differences.sort()
    return [differences[int(0.025 * samples)], differences[int(0.975 * samples)]]


def bundle_hits(
    bundles: list[Any], scores: dict[str, list[Any]], weight: Any, candidate_limit: int, policy: dict[str, Any]
) -> tuple[list[int], list[int]]:
    from scripts.evaluate.run_c3_cardinality_robust_dev_v1 import (
        choose_with_policy,
        independent_choice,
    )

    baseline_hits, method_hits = [], []
    for bundle in bundles:
        baseline = independent_choice(bundle, scores)
        choice, _ = choose_with_policy(
            bundle,
            scores,
            weight,
            candidate_limit,
            alpha=float(policy["alpha"]),
            max_regret_ratio=float(policy["max_regret_ratio"]),
            max_source_count_delta=int(policy["max_source_count_delta"]),
        )
        baseline_hits.append(
            int(all(scores[group.group_id][index].gold for group, index in zip(bundle.groups, baseline, strict=True)))
        )
        method_hits.append(
            int(all(scores[group.group_id][index].gold for group, index in zip(bundle.groups, choice, strict=True)))
        )
    return baseline_hits, method_hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--group-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from provtimerag.routing.multitask_model import MultiTaskProvenanceRouter
    from scripts.evaluate.run_c3_publisher_structured_v3 import publisher_router_scores

    policy_id, policy = selected_configuration(args.frozen_policy)
    train = external.load_bundles(args.train_input, None)
    holdout = external.load_bundles(args.input, None)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer", local_files_only=True)
    router = MultiTaskProvenanceRouter.from_pretrained(args.checkpoint / "backbone", dropout=0.1)
    router.heads.load_state_dict(
        torch.load(args.checkpoint / "router_heads.pt", map_location="cpu", weights_only=True)
    )
    router.to(device=device, dtype=torch.float32).eval()

    def score(rows: list[Any]) -> dict[str, list[Any]]:
        return publisher_router_scores(
            router,
            tokenizer,
            rows,
            batch_size=args.group_batch_size,
            max_length=args.max_length,
            device=device,
        )

    train_scores = score(train)
    holdout_scores = score(holdout)
    weight = external.fit_assignment_head(
        train, train_scores, args.candidate_limit, args.epochs, args.seed
    )
    independent = report_policy(holdout, holdout_scores, weight, args.candidate_limit, None)
    robust = report_policy(holdout, holdout_scores, weight, args.candidate_limit, policy)
    baseline_hits, robust_hits = bundle_hits(
        holdout, holdout_scores, weight, args.candidate_limit, policy
    )
    improvements = sum(a == 0 and b == 1 for a, b in zip(baseline_hits, robust_hits, strict=True))
    regressions = sum(a == 1 and b == 0 for a, b in zip(baseline_hits, robust_hits, strict=True))
    result = {
        "status": "complete",
        "method": "frozen_cardinality_shift_robust_c3_external_v1",
        "evaluation_contract": "untouched_external_test_no_policy_selection",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "frozen_policy_id": policy_id,
        "frozen_policy": policy,
        "input_sha256": {
            "train": sha256_file(args.train_input),
            "holdout": sha256_file(args.input),
            "frozen_policy": sha256_file(args.frozen_policy),
        },
        "counts": {
            "train_bundles": len(train),
            "holdout_bundles": len(holdout),
            "holdout_groups": sum(len(bundle.groups) for bundle in holdout),
        },
        "feature_names": list(external.FEATURE_NAMES),
        "assignment_weights": weight.tolist(),
        "metrics": {"independent": independent, "frozen_robust": robust},
        "paired_significance": {
            "improvements": improvements,
            "regressions": regressions,
            "exact_two_sided_mcnemar_p": exact_two_sided_binomial(improvements, regressions),
            "bundle_exact_delta": robust["bundle_exact_match"] - independent["bundle_exact_match"],
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                baseline_hits, robust_hits, args.seed
            ),
        },
        "policy": [
            "The decoder policy was frozen on in-domain calibration plus AVerImaTeC v2 development before this holdout was scored.",
            "No threshold, feature, checkpoint, or policy is selected on this holdout.",
            "The assignment head is fit on publisher train bundles only.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
