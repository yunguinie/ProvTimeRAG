from __future__ import annotations

from collections import Counter

from scripts.prepare.build_averimatec_publisher_source_swap_v1 import build_bundles
from scripts.prepare.repair_averimatec_donor_balance_v2 import repair_bundles


def row(index: int) -> dict:
    return {
        "claim_text": f"Claim {index}",
        "questions": [
            {
                "question": f"Question {index}-a?",
                "answers": [
                    {
                        "answer_text": f"Answer {index}-a",
                        "source_url": f"https://publisher-{index}-a.example/story",
                    }
                ],
            },
            {
                "question": f"Question {index}-b?",
                "answers": [
                    {
                        "answer_text": f"Answer {index}-b",
                        "source_url": f"https://publisher-{index}-b.example/story",
                    }
                ],
            },
        ],
    }


def test_balanced_repair_uses_every_positive_donor_once() -> None:
    protected = {"queries": set(), "urls": set(), "texts": set()}
    v1, _ = build_bundles([row(index) for index in range(4)], protected)
    repaired, audit = repair_bundles(v1)
    positives = Counter()
    negatives = Counter()
    for bundle in repaired:
        for group in bundle.groups:
            positive = next(c for c in group.candidates if c.gold_route)
            negative = next(c for c in group.candidates if not c.gold_route)
            positives[positive.evidence.source_id] += 1
            negatives[negative.evidence.source_id] += 1
            assert positive.evidence.source_id != negative.evidence.source_id
            assert positive.evidence.text == negative.evidence.text
    assert positives == negatives
    assert audit["all_donor_candidates_used_once"] is True
    assert audit["positive_negative_source_histograms_equal"] is True
