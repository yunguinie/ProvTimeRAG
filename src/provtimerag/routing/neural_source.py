"""Pairwise neural source-routing utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch.nn import functional

from provtimerag.data import ClaimBundle, ClaimRouteGroup, EvidenceRecord


@dataclass(frozen=True)
class SourcePair:
    group_id: str
    request_text: str
    positive_text: str
    negative_text: str
    positive_source: str
    negative_source: str


def request_text(group: ClaimRouteGroup) -> str:
    required = ", ".join(group.claim.required_source_roles) or "unspecified"
    return (
        f"Question: {group.query.text}\n"
        f"Claim need: {group.claim.text}\n"
        f"Required source role: {required}"
    )


def candidate_text(evidence: EvidenceRecord) -> str:
    media = evidence.metadata.get("source_medium") or "unspecified"
    return (
        f"Evidence: {evidence.text}\n"
        f"Source identity: {evidence.source_id}\n"
        f"Source role: {evidence.source_role}\n"
        f"Source medium: {media}"
    )


def source_pair(bundle: ClaimBundle) -> SourcePair:
    if len(bundle.groups) != 1:
        raise ValueError("source pair bundles must contain one route group")
    group = bundle.groups[0]
    negatives = [
        candidate
        for candidate in group.candidates
        if not candidate.gold_route
        and candidate.risk_type.value == "wrong_attribution"
    ]
    if len(negatives) != 1:
        raise ValueError("expected exactly one wrong-attribution candidate")
    negative = negatives[0]
    original_source = negative.evidence.metadata.get("original_source_id")
    positives = [
        candidate
        for candidate in group.candidates
        if candidate.gold_route
        and (
            original_source is None
            or candidate.evidence.source_id == original_source
        )
    ]
    if not positives:
        raise ValueError("no matching positive source route")
    positive = positives[0]
    return SourcePair(
        group_id=group.group_id,
        request_text=request_text(group),
        positive_text=candidate_text(positive.evidence),
        negative_text=candidate_text(negative.evidence),
        positive_source=positive.evidence.source_id,
        negative_source=negative.evidence.source_id,
    )


def deterministic_pairs(
    bundles: list[ClaimBundle],
    limit: int | None,
) -> list[SourcePair]:
    ordered = sorted(
        bundles,
        key=lambda bundle: hashlib.sha256(
            bundle.bundle_id.encode()
        ).hexdigest(),
    )
    selected = ordered if limit is None else ordered[:limit]
    return [source_pair(bundle) for bundle in selected]


def pairwise_rank_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive and negative scores must have equal shape")
    return -functional.logsigmoid(
        positive_scores - negative_scores
    ).mean()
