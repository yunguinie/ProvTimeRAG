"""Leakage-safe utilities for fusing heterogeneous candidate scores."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ScoreNormalizer:
    """Training-split statistics used to standardize one scoring channel."""

    mean: float
    scale: float

    def transform(self, value: float) -> float:
        return (float(value) - self.mean) / self.scale


def fit_normalizer(values: Iterable[float], *, minimum_scale: float = 1e-6) -> ScoreNormalizer:
    """Fit a population-standard-deviation normalizer on training scores only."""

    observed = [float(value) for value in values]
    if not observed:
        raise ValueError("cannot fit a score normalizer without observations")
    if not all(isfinite(value) for value in observed):
        raise ValueError("score normalizer received a non-finite value")
    mean = sum(observed) / len(observed)
    variance = sum((value - mean) ** 2 for value in observed) / len(observed)
    return ScoreNormalizer(mean=mean, scale=max(variance**0.5, minimum_scale))


def convex_fuse(
    router_scores: Sequence[float],
    lexical_scores: Sequence[float],
    *,
    router_normalizer: ScoreNormalizer,
    lexical_normalizer: ScoreNormalizer,
    router_weight: float,
) -> list[float]:
    """Fuse aligned Router and lexical scores after train-only normalization."""

    if len(router_scores) != len(lexical_scores):
        raise ValueError("router and lexical score sequences must be aligned")
    if not 0.0 <= router_weight <= 1.0:
        raise ValueError("router_weight must be between zero and one")
    lexical_weight = 1.0 - router_weight
    return [
        router_weight * router_normalizer.transform(router)
        + lexical_weight * lexical_normalizer.transform(lexical)
        for router, lexical in zip(router_scores, lexical_scores, strict=True)
    ]


def minmax_scale(values: Sequence[float]) -> list[float]:
    """Scale one candidate group to [0, 1] without cross-group assumptions."""

    observed = [float(value) for value in values]
    if not observed:
        return []
    if not all(isfinite(value) for value in observed):
        raise ValueError("score scaling received a non-finite value")
    lower, upper = min(observed), max(observed)
    if upper == lower:
        return [0.0 for _ in observed]
    return [(value - lower) / (upper - lower) for value in observed]


def groupwise_convex_fuse(
    router_scores: Sequence[float],
    lexical_scores: Sequence[float],
    *,
    router_weight: float,
) -> list[float]:
    """Fuse scores after scaling each candidate group independently."""

    if len(router_scores) != len(lexical_scores):
        raise ValueError("router and lexical score sequences must be aligned")
    if not 0.0 <= router_weight <= 1.0:
        raise ValueError("router_weight must be between zero and one")
    router = minmax_scale(router_scores)
    lexical = minmax_scale(lexical_scores)
    lexical_weight = 1.0 - router_weight
    return [
        router_weight * router_value + lexical_weight * lexical_value
        for router_value, lexical_value in zip(router, lexical, strict=True)
    ]
