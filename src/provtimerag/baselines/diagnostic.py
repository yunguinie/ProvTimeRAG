"""Dependency-free diagnostic rankers for the Pilot benchmark."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from provtimerag.data.models import ClaimRouteGroup, EvidenceRecord

TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


@dataclass(frozen=True)
class RankedGroup:
    evidence_ids: tuple[str, ...]
    scores: tuple[float, ...]
    abstain: bool


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def _tie_key(group_id: str, evidence_id: str) -> str:
    return hashlib.sha256(f"{group_id}\0{evidence_id}".encode()).hexdigest()


def lexical_scores(group: ClaimRouteGroup) -> dict[str, float]:
    """Compute BM25-style scores within one pre-retrieved candidate list."""

    query_terms = tokenize(f"{group.query.text} {group.claim.text}")
    documents = [tokenize(item.evidence.text) for item in group.candidates]
    document_count = len(documents)
    average_length = sum(map(len, documents)) / max(document_count, 1)
    document_frequency = Counter(
        term for document in documents for term in set(document)
    )
    scores: dict[str, float] = {}
    for candidate, document in zip(group.candidates, documents, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(document) / max(average_length, 1.0)
            )
            score += inverse_document_frequency * frequency * 2.2 / denominator
        scores[candidate.evidence.evidence_id] = score
    return scores


def _time_compatibility(target: datetime | None, evidence: EvidenceRecord) -> float:
    if target is None:
        return 0.0
    if evidence.valid_from and target < evidence.valid_from:
        return -1.0
    if evidence.valid_to and target > evidence.valid_to:
        return -1.0
    return 1.0 if evidence.valid_from or evidence.valid_to else 0.0


def metadata_scores(group: ClaimRouteGroup) -> dict[str, float]:
    """Add observable time, recency, and requested-role features to lexical score."""

    scores = lexical_scores(group)
    publication_times = [
        item.evidence.publication_time
        for item in group.candidates
        if item.evidence.publication_time is not None
    ]
    newest = max(publication_times) if publication_times else None
    required_roles = set(group.claim.required_source_roles)
    for candidate in group.candidates:
        evidence = candidate.evidence
        score = scores[evidence.evidence_id]
        score += 2.0 * _time_compatibility(group.claim.target_time, evidence)
        if newest and evidence.publication_time == newest:
            score += 0.5
        if required_roles and evidence.source_role in required_roles:
            score += 0.5
        scores[evidence.evidence_id] = score
    return scores


def rank_scores(
    group: ClaimRouteGroup,
    scores: dict[str, float],
    *,
    abstain_threshold: float | None = None,
) -> RankedGroup:
    ordered = sorted(
        scores,
        key=lambda evidence_id: (
            -scores[evidence_id],
            _tie_key(group.group_id, evidence_id),
        ),
    )
    top_score = scores[ordered[0]] if ordered else float("-inf")
    return RankedGroup(
        evidence_ids=tuple(ordered),
        scores=tuple(scores[evidence_id] for evidence_id in ordered),
        abstain=abstain_threshold is not None and top_score <= abstain_threshold,
    )


def rank_lexical(
    group: ClaimRouteGroup, *, selective: bool = False
) -> RankedGroup:
    return rank_scores(
        group,
        lexical_scores(group),
        abstain_threshold=0.0 if selective else None,
    )


def rank_metadata(
    group: ClaimRouteGroup, *, selective: bool = False
) -> RankedGroup:
    return rank_scores(
        group,
        metadata_scores(group),
        abstain_threshold=0.0 if selective else None,
    )


def rank_random(group: ClaimRouteGroup) -> RankedGroup:
    scores = {
        item.evidence.evidence_id: int(
            _tie_key(group.group_id, item.evidence.evidence_id)[:12], 16
        )
        / float(16**12)
        for item in group.candidates
    }
    return rank_scores(group, scores)


def candidate_texts(group: ClaimRouteGroup) -> Sequence[str]:
    """Return observable candidate text for debugging without labels."""

    return tuple(item.evidence.text for item in group.candidates)
