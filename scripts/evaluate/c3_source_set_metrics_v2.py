"""Complete source-set metrics for frozen C3 decoding."""

from __future__ import annotations

from typing import Any

from scripts.evaluate import run_c3_cardinality_robust_dev_v1 as robust


def report_policy(
    bundles: list[Any],
    scores: dict[str, list[Any]],
    weight: Any,
    candidate_limit: int,
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    report = robust.report_policy(bundles, scores, weight, candidate_limit, policy)
    exact = count_mismatch = 0
    precision_sum = recall_sum = jaccard_sum = 0.0
    for bundle in bundles:
        if policy is None:
            choice = robust.independent_choice(bundle, scores)
        else:
            choice, _ = robust.choose_with_policy(
                bundle,
                scores,
                weight,
                candidate_limit,
                alpha=float(policy["alpha"]),
                max_regret_ratio=float(policy["max_regret_ratio"]),
                max_source_count_delta=int(policy["max_source_count_delta"]),
            )
        selected = [
            scores[group.group_id][index]
            for group, index in zip(bundle.groups, choice, strict=True)
        ]
        gold_sources = {
            candidate.evidence.source_id
            for group in bundle.groups
            for candidate in group.candidates
            if candidate.gold_route
        }
        predicted_sources = {item.evidence.source_id for item in selected}
        intersection = gold_sources & predicted_sources
        union = gold_sources | predicted_sources
        exact += int(gold_sources == predicted_sources)
        count_mismatch += int(len(gold_sources) != len(predicted_sources))
        precision_sum += len(intersection) / len(predicted_sources)
        recall_sum += len(intersection) / len(gold_sources)
        jaccard_sum += len(intersection) / len(union)
    count = len(bundles)
    report.update(
        {
            "source_set_exact_match": exact / count,
            "source_set_jaccard": jaccard_sum / count,
            "source_set_precision": precision_sum / count,
            "source_set_recall": recall_sum / count,
            "source_count_mismatch_rate": count_mismatch / count,
            "binary_multi_source_error_rate": report.pop("total_source_error"),
        }
    )
    return report
