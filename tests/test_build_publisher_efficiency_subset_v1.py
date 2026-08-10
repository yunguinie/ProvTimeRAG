from scripts.prepare.build_publisher_efficiency_subset_v1 import stable_subset


def test_stable_subset_is_input_order_independent():
    rows = [{"bundle_id": value} for value in ("b", "a", "c")]
    assert stable_subset(rows, 2) == stable_subset(list(reversed(rows)), 2)


def test_stable_subset_rejects_duplicate_ids():
    try:
        stable_subset([{"bundle_id": "a"}, {"bundle_id": "a"}], 1)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate ids should fail")
