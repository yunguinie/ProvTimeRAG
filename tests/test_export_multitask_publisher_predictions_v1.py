from scripts.evaluate.export_multitask_publisher_predictions_v1 import frequency_bucket


def test_frequency_buckets_are_disjoint_and_exhaustive() -> None:
    assert frequency_bucket(0) == "unseen"
    assert frequency_bucket(1) == "tail_1_5"
    assert frequency_bucket(5) == "tail_1_5"
    assert frequency_bucket(6) == "mid_6_20"
    assert frequency_bucket(20) == "mid_6_20"
    assert frequency_bucket(21) == "head_21_plus"
