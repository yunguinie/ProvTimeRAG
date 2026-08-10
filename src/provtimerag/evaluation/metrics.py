"""Routing metrics independent of model framework."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from provtimerag.data.models import ClaimBundle


@dataclass(frozen=True)
class RoutingMetrics:
    recall_at_k: dict[int, float]
    mean_reciprocal_rank: float
    top1_accuracy: float
    abstention_accuracy: float
    bundle_exact_match: float


def evaluate_rankings(
    bundles: Iterable[ClaimBundle],
    rankings: Mapping[str, Sequence[str]],
    abstentions: Mapping[str, bool],
    ks: tuple[int, ...] = (1, 5),
) -> RoutingMetrics:
    materialized = list(bundles)
    groups = [group for bundle in materialized for group in bundle.groups]
    if not groups:
        raise ValueError("at least one claim route group is required")
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    missing_ranks = [group.group_id for group in groups if group.group_id not in rankings]
    missing_abstain = [group.group_id for group in groups if group.group_id not in abstentions]
    if missing_ranks or missing_abstain:
        raise ValueError(
            f"missing predictions: rankings={missing_ranks}, abstentions={missing_abstain}"
        )

    recall_hits = {k: 0 for k in ks}
    rr_sum = top1_hits = abstention_hits = 0.0
    group_correct: dict[str, bool] = {}
    for group in groups:
        ranking = list(rankings[group.group_id])
        if len(ranking) != len(set(ranking)):
            raise ValueError(f"ranking for '{group.group_id}' has duplicate evidence IDs")
        candidate_ids = {item.evidence.evidence_id for item in group.candidates}
        unknown = set(ranking) - candidate_ids
        if unknown:
            raise ValueError(
                f"ranking for '{group.group_id}' contains unknown evidence IDs: {sorted(unknown)}"
            )
        predicted_abstain = abstentions[group.group_id]
        abstention_hits += int(predicted_abstain == group.should_abstain)
        gold = group.gold_evidence_ids
        if group.should_abstain:
            group_correct[group.group_id] = predicted_abstain
            continue
        rank = next((i for i, evidence_id in enumerate(ranking, 1) if evidence_id in gold), 0)
        rr_sum += 1.0 / rank if rank else 0.0
        for k in ks:
            recall_hits[k] += int(bool(set(ranking[:k]) & gold))
        correct = bool(ranking and ranking[0] in gold and not predicted_abstain)
        top1_hits += int(correct)
        group_correct[group.group_id] = correct

    routed_count = sum(not group.should_abstain for group in groups)
    denominator = max(routed_count, 1)
    exact = sum(
        all(group_correct[group.group_id] for group in bundle.groups) for bundle in materialized
    )
    return RoutingMetrics(
        recall_at_k={k: recall_hits[k] / denominator for k in ks},
        mean_reciprocal_rank=rr_sum / denominator,
        top1_accuracy=top1_hits / denominator,
        abstention_accuracy=abstention_hits / len(groups),
        bundle_exact_match=exact / len(materialized),
    )
