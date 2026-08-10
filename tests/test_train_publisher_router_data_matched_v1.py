import pytest

from provtimerag.routing.multitask_data import RouterGroupExample, TASK_SOURCE
from scripts.train.train_publisher_router_data_matched_v1 import select_source_epoch


def example(group_id: str) -> RouterGroupExample:
    return RouterGroupExample(
        group_id=group_id,
        task=TASK_SOURCE,
        request_text="request",
        candidate_texts=("candidate",),
        candidate_source_ids=("source",),
        gold_route=(True,),
        source_match=(True,),
        temporal_valid=(True,),
        version_valid=(True,),
        should_abstain=False,
    )


def test_data_matched_epoch_is_unique_and_exact() -> None:
    rows = [example("g1"), example("g2"), example("g3")]
    selected = select_source_epoch(rows, source_groups=3, seed=42)
    assert len(selected) == 3
    assert {row.group_id for row in selected} == {"g1", "g2", "g3"}


def test_data_matched_epoch_rejects_oversampling() -> None:
    with pytest.raises(ValueError, match="exactly 3 unique"):
        select_source_epoch([example("g1"), example("g2")], source_groups=3, seed=42)
