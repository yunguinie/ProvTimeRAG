from __future__ import annotations

from provtimerag.routing.multitask_data import RouterGroupExample
from scripts.evaluate.evaluate_multitask_publisher_router_v3 import (
    gold_sources,
    has_publisher_url_contract,
)


def example(*, texts: tuple[str, ...], sources: tuple[str, ...]) -> RouterGroupExample:
    return RouterGroupExample(
        group_id="g1",
        task="source",
        request_text="request",
        candidate_texts=texts,
        candidate_source_ids=sources,
        gold_route=(True, False),
        source_match=(True, False),
        temporal_valid=(True, True),
        version_valid=(True, True),
        should_abstain=False,
    )


def test_gold_sources_uses_only_positive_candidates() -> None:
    rows = [
        example(
            texts=("Source URL: a\nSource domain: a", "Source URL: b\nSource domain: b"),
            sources=("publisher.example", "donor.example"),
        )
    ]
    assert gold_sources(rows) == {"publisher.example"}


def test_publisher_url_contract_rejects_legacy_candidate_text() -> None:
    valid = example(
        texts=("Source URL: a\nSource domain: a", "Source URL: b\nSource domain: b"),
        sources=("a", "b"),
    )
    invalid = example(texts=("Evidence: a", "Evidence: b"), sources=("a", "b"))
    assert has_publisher_url_contract([valid]) is True
    assert has_publisher_url_contract([invalid]) is False
