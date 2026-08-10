"""Baselines used for diagnostics and paper comparisons."""

from provtimerag.baselines.diagnostic import (
    RankedGroup,
    lexical_scores,
    metadata_scores,
    rank_lexical,
    rank_metadata,
    rank_random,
)

__all__ = [
    "RankedGroup",
    "lexical_scores",
    "metadata_scores",
    "rank_lexical",
    "rank_metadata",
    "rank_random",
]
