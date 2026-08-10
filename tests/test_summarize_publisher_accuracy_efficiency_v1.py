from scripts.evaluate.summarize_publisher_accuracy_efficiency_v1 import accuracy


def test_accuracy_reads_multiseed_c2_and_frozen_baseline():
    report = {
        "c2_three_seed_summary": {
            "group_top1_mean": 0.9,
            "group_top1_sample_std": 0.01,
        },
        "baselines": {"minilm": {"group": {"accuracy": 0.5}}},
    }
    assert accuracy(report, "c2_multitask") == (0.9, 0.01)
    assert accuracy(report, "minilm") == (0.5, None)
