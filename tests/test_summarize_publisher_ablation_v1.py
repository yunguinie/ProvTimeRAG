from scripts.evaluate.summarize_publisher_ablation_v1 import summarize


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
