from scripts.evaluate.compare_publisher_router_variants_v1 import (
    bundle_hits,
    exact_mcnemar,
    paired_report,
)


def test_exact_mcnemar_is_symmetric() -> None:
    assert exact_mcnemar(9, 2) == exact_mcnemar(2, 9)
    assert exact_mcnemar(0, 0) == 1.0


def test_paired_report_tracks_direction() -> None:
    first = {"a": True, "b": True, "c": False, "d": True}
    second = {"a": True, "b": False, "c": False, "d": False}
    report = paired_report(first, second)
    assert report["first_accuracy"] == 0.75
    assert report["second_accuracy"] == 0.25
    assert report["delta"] == 0.5
    assert report["improvements"] == 2
    assert report["regressions"] == 0


def test_bundle_hits_requires_complete_population() -> None:
    bundles = {"b1": ["g1", "g2"], "b2": ["g3"]}
    assert bundle_hits({"g1": True, "g2": False, "g3": True}, bundles) == {
        "b1": False,
        "b2": True,
    }
