from __future__ import annotations

from datetime import date

from provtimerag.data import (
    CandidateRoute,
    ClaimRouteGroup,
    EvidenceRecord,
    QueryRecord,
    RiskType,
    SupportLabel,
)
from scripts.evaluate.run_c3_strong_cross_encoder_baseline_v1 import (
    observable_pair,
    ranking_from_scores,
)


def group() -> ClaimRouteGroup:
    query = QueryRecord(query_id="q", text="Who published the report?")
    evidence = []
    for index, domain in enumerate(("alpha.example", "beta.example")):
        evidence.append(
            EvidenceRecord(
                evidence_id=f"e{index}",
                document_id=f"d{index}",
                source_id=domain,
                source_role="publisher",
                text="identical evidence text",
                publication_time=date(2024, 1, 1),
                valid_from=date(2024, 1, 1),
                version_id="v1",
                metadata={"source_url": f"https://{domain}/item"},
            )
        )
    return ClaimRouteGroup.model_validate(
        {
            "group_id": "g",
            "dataset": "unit",
            "query": query.model_dump(mode="json"),
            "claim": {
                "claim_id": "c",
                "query_id": "q",
                "claim_type": "publisher_attribution",
                "text": "publisher attribution",
                "required_source_roles": ["publisher"],
                "temporal_scope": "current",
            },
            "candidates": [
                CandidateRoute(
                    evidence=item,
                    gold_route=index == 0,
                    support_label=SupportLabel.SUPPORTED,
                    source_match=index == 0,
                    temporal_valid=True,
                    version_valid=True,
                    completeness=1.0,
                    risk_type=RiskType.NONE if index == 0 else RiskType.WRONG_ATTRIBUTION,
                ).model_dump(mode="json")
                for index, item in enumerate(evidence)
            ],
            "should_abstain": False,
        }
    )


def test_input_views_expose_only_declared_fields() -> None:
    item = group()
    _, content = observable_pair(item, 0, input_view="content_only")
    _, visible = observable_pair(item, 0, input_view="publisher_visible")
    assert "alpha.example" not in content
    assert "Source URL: https://alpha.example/item" in visible
    assert "Source domain: alpha.example" in visible


def test_ranking_uses_scores_and_label_independent_ties() -> None:
    item = group()
    assert ranking_from_scores(item, [0.1, 0.9])[0] == "e1"
    first = ranking_from_scores(item, [0.5, 0.5])
    second = ranking_from_scores(item, [0.5, 0.5])
    assert first == second
    assert set(first) == {"e0", "e1"}




