"""Utilities for text-only Cross-Encoder routing baselines."""

from __future__ import annotations

from collections.abc import Sequence

from provtimerag.baselines.diagnostic import RankedGroup, rank_scores
from provtimerag.data.models import ClaimRouteGroup


def cross_encoder_pairs(group: ClaimRouteGroup) -> list[tuple[str, str]]:
    """Build observable text pairs without labels or provenance metadata."""

    request = f"Question: {group.query.text}\nClaim need: {group.claim.text}"
    return [(request, candidate.evidence.text) for candidate in group.candidates]


def rank_cross_encoder_scores(
    group: ClaimRouteGroup,
    scores: Sequence[float],
) -> RankedGroup:
    if len(scores) != len(group.candidates):
        raise ValueError("one Cross-Encoder score is required for every candidate")
    mapped = {
        candidate.evidence.evidence_id: float(score)
        for candidate, score in zip(group.candidates, scores, strict=True)
    }
    return rank_scores(group, mapped)
