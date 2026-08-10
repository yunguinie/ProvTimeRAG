"""Label-independent C3 tie-breaking using exactly model-visible evidence text."""

from __future__ import annotations

import hashlib
from typing import Any

from provtimerag.routing.multitask_data import candidate_text


def stable_visible_tie_key(group_id: str, evidence: Any) -> str:
    visible = candidate_text(evidence, include_source_url=True)
    return hashlib.sha256(f"{group_id}\0{visible}".encode("utf-8")).hexdigest()


def best_candidate_index(group_id: str, items: list[Any]) -> int:
    """Match publisher-router evaluation without consulting labels or IDs."""

    return min(
        range(len(items)),
        key=lambda index: (
            -items[index].route_score,
            stable_visible_tie_key(group_id, items[index].evidence),
        ),
    )


def install() -> None:
    """Install the corrected selector in the shared C3 evaluation module."""

    from scripts.evaluate import run_c3_external_baselines_v1 as external

    external.best_candidate_index = best_candidate_index
