from __future__ import annotations

import torch

from provtimerag.routing.multitask_data import RouterGroupExample
from scripts.train.train_multitask_publisher_router_v3 import stable_best_index


def example(candidate_texts: tuple[str, ...]) -> RouterGroupExample:
    return RouterGroupExample(
        group_id="g1",
        task="source",
        request_text="request",
        candidate_texts=candidate_texts,
        candidate_source_ids=("a.example", "b.example"),
        gold_route=(True, False),
        source_match=(True, False),
        temporal_valid=(True, True),
        version_valid=(True, True),
        should_abstain=False,
    )


def test_stable_tie_break_is_invariant_to_candidate_order() -> None:
    first = example(("candidate-a", "candidate-b"))
    selected_first, tied_first = stable_best_index(first, torch.tensor([1.0, 1.0]))
    selected_text = first.candidate_texts[selected_first]

    second = example(("candidate-b", "candidate-a"))
    selected_second, tied_second = stable_best_index(second, torch.tensor([1.0, 1.0]))

    assert tied_first is True
    assert tied_second is True
    assert second.candidate_texts[selected_second] == selected_text
