from scripts.evaluate.summarize_publisher_ablation_v1 import (
    exposed_training_groups,
    false_abstention_rate,
    summarize,
)


def test_summarize_reports_mean_and_sample_std():
    rows = []
    for value in (1.0, 2.0, 3.0):
        row = {
            "source_url_included": True,
            "training_groups": value,
            "optimizer_steps": value,
            "training_seconds": value,
            "dev_publisher_top1": value,
            "dev_temporal_top1": value,
            "dev_insufficient_f1": value,
            "clean_blind_top1": value,
            "finfact_top1": value,
            "finfact_false_abstention_rate": value,
        }
        rows.append(row)
    result = summarize(rows)
    assert result["clean_blind_top1"]["mean"] == 2.0
    assert result["clean_blind_top1"]["sample_std"] == 1.0

def test_exposed_training_groups_supports_both_report_contracts():
    assert exposed_training_groups({}, {"training_groups": 7}) == 7
    assert exposed_training_groups({"groups_per_task": 7}, {}) == 21


def test_false_abstention_rate_reads_nested_confusion():
    abstention = {"support": 10, "confusion": {"fp": 3}}
    assert false_abstention_rate(abstention) == 0.3
