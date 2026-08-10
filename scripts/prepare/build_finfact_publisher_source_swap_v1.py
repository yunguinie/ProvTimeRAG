"""Build a frozen Fin-Fact publisher Source-Swap external benchmark.

Only direct, single-publisher evidence links are retained.  Archive access
hosts, URL shorteners, and fact-check-site self citations are excluded.  A
deterministic perfect matching assigns every positive publisher exactly once
as a negative donor, so label frequencies cannot be learned from publisher
frequency.  Protected claim, URL, and evidence-text overlap removes the whole
claim before matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from provtimerag.data import (
    CandidateRoute,
    ClaimBundle,
    ClaimRecord,
    ClaimRouteGroup,
    EvidenceRecord,
    QueryRecord,
    RiskType,
    SupportLabel,
    TemporalScope,
    validate_dataset,
    write_bundles_jsonl,
)
from provtimerag.data.adapters.common import stable_id
from scripts.prepare.build_averimatec_publisher_source_swap_v1 import (
    canonical_publisher_url,
    normalize_text,
    protected_signatures,
    sha256_file,
    text_fingerprint,
)


EXCLUDED_HOSTS = {
    "archive.fo",
    "archive.is",
    "archive.ph",
    "archive.vn",
    "factcheck.afp.com",
    "factcheck.org",
    "fullfact.org",
    "leadstories.com",
    "perma.cc",
    "politifact.com",
    "snopes.com",
    "t.co",
    "web.archive.org",
}


def excluded_host(host: str) -> bool:
    return any(host == value or host.endswith(f".{value}") for value in EXCLUDED_HOSTS)


def stable_hash(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def stable_candidate_order(candidate: CandidateRoute) -> str:
    evidence = candidate.evidence
    return stable_hash(
        "finfact-candidate-order-v1",
        normalize_text(evidence.text),
        str(evidence.metadata.get("source_url") or ""),
    )


def usable_evidence(record: dict[str, Any], row_index: int, audit: Counter[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for evidence_index, item in enumerate(record.get("evidence") or []):
        if not isinstance(item, dict):
            audit["evidence_not_object"] += 1
            continue
        sentence = " ".join(str(item.get("sentence") or "").split())
        if not sentence:
            audit["evidence_missing_sentence"] += 1
            continue
        urls_by_source: dict[str, list[str]] = {}
        for raw_url in item.get("hrefs") or []:
            url, source_id, method = canonical_publisher_url(raw_url)
            if not url or not source_id:
                audit[f"url_excluded_{method}"] += 1
                continue
            if excluded_host(source_id):
                audit["url_excluded_nonpublisher_or_unresolved"] += 1
                continue
            urls_by_source.setdefault(source_id, []).append(url)
        if len(urls_by_source) != 1:
            audit[
                "evidence_excluded_no_usable_publisher"
                if not urls_by_source
                else "evidence_excluded_multiple_publishers"
            ] += 1
            continue
        source_id, urls = next(iter(urls_by_source.items()))
        groups.append(
            {
                "row_index": row_index,
                "evidence_index": evidence_index,
                "text": sentence,
                "url": sorted(set(urls))[0],
                "source_id": source_id,
            }
        )
    return groups


def perfect_matching(groups: list[dict[str, Any]]) -> dict[int, int]:
    adjacency: list[list[int]] = []
    for left, group in enumerate(groups):
        eligible = [
            right
            for right, donor in enumerate(groups)
            if right != left and donor["source_id"] != group["source_id"]
        ]
        eligible.sort(
            key=lambda right: stable_hash(
                "finfact-balanced-donor-v1",
                str(group["row_index"]),
                str(group["evidence_index"]),
                str(groups[right]["row_index"]),
                str(groups[right]["evidence_index"]),
                groups[right]["url"],
            )
        )
        if not eligible:
            raise RuntimeError(f"no legal donor for Fin-Fact group {left}")
        adjacency.append(eligible)

    right_to_left: dict[int, int] = {}

    def augment(left: int, visited: set[int]) -> bool:
        for right in adjacency[left]:
            if right in visited:
                continue
            visited.add(right)
            previous = right_to_left.get(right)
            if previous is None or augment(previous, visited):
                right_to_left[right] = left
                return True
        return False

    order = sorted(
        range(len(groups)),
        key=lambda index: stable_hash(
            "finfact-balanced-left-v1",
            str(groups[index]["row_index"]),
            str(groups[index]["evidence_index"]),
        ),
    )
    for left in order:
        if not augment(left, set()):
            raise RuntimeError(f"no perfect Fin-Fact donor matching at group {left}")
    if len(right_to_left) != len(groups):
        raise RuntimeError("incomplete Fin-Fact donor matching")
    return {left: right for right, left in right_to_left.items()}


def build_bundles(
    rows: list[dict[str, Any]], protected: dict[str, set[str]]
) -> tuple[list[ClaimBundle], dict[str, Any]]:
    audit: Counter[str] = Counter()
    prepared: list[dict[str, Any]] = []
    for row_index, record in enumerate(rows):
        claim = " ".join(str(record.get("claim") or "").split())
        if not claim:
            audit["claims_missing_text"] += 1
            continue
        groups = usable_evidence(record, row_index, audit)
        if len(groups) < 2:
            audit["claims_excluded_fewer_than_two_groups"] += 1
            continue
        signatures = {
            "queries": {normalize_text(claim)},
            "urls": {group["url"] for group in groups},
            "texts": {text_fingerprint(group["text"]) for group in groups},
        }
        overlap = {name: signatures[name] & protected[name] for name in signatures}
        if any(overlap.values()):
            audit["claims_excluded_protected_overlap"] += 1
            for name, values in overlap.items():
                audit[f"protected_{name}_overlap_items"] += len(values)
            continue
        prepared.append(
            {"row_index": row_index, "record": record, "claim": claim, "groups": groups}
        )

    flat = [group for item in prepared for group in item["groups"]]
    if not flat:
        raise RuntimeError("no eligible Fin-Fact groups remain")
    matching = perfect_matching(flat)
    location = {
        (group["row_index"], group["evidence_index"]): index
        for index, group in enumerate(flat)
    }
    bundles: list[ClaimBundle] = []
    for item in prepared:
        row_index = item["row_index"]
        record = item["record"]
        query_id = stable_id("finfact-query-v1", f"{row_index}:{item['claim']}")
        query = QueryRecord(
            query_id=query_id,
            text=item["claim"],
            metadata={
                "dataset": "finfact",
                "label": record.get("label"),
                "posted": record.get("posted"),
                "fact_check_url": record.get("url"),
                "text_evidence_only": True,
                "images_loaded": False,
            },
        )
        route_groups = []
        for group in item["groups"]:
            flat_index = location[(row_index, group["evidence_index"])]
            donor = flat[matching[flat_index]]
            claim_id = stable_id(
                "finfact-subclaim-v1", f"{query_id}:{group['evidence_index']}"
            )
            positive_evidence = EvidenceRecord(
                evidence_id=stable_id(
                    "finfact-positive-v1", f"{claim_id}:{group['url']}:{group['text']}"
                ),
                text=group["text"],
                source_id=group["source_id"],
                source_role="publisher_citation_target",
                document_id=stable_id("finfact-publisher-document-v1", group["url"]),
                version_id=stable_id("finfact-unversioned-v1", group["url"]),
                metadata={
                    "source_url": group["url"],
                    "publisher_identity_contract": "direct_citation_target_v1",
                },
            )
            positive = CandidateRoute(
                evidence=positive_evidence,
                gold_route=True,
                support_label=SupportLabel.SUPPORTED,
                source_match=True,
                temporal_valid=True,
                version_valid=True,
                completeness=1.0,
                risk_type=RiskType.NONE,
            )
            negative_evidence = positive_evidence.model_copy(
                update={
                    "evidence_id": stable_id(
                        "finfact-source-swap-v1",
                        f"{claim_id}:{donor['url']}:{group['text']}",
                    ),
                    "source_id": donor["source_id"],
                    "document_id": stable_id("finfact-publisher-document-v1", donor["url"]),
                    "version_id": stable_id("finfact-unversioned-v1", donor["url"]),
                    "metadata": {
                        "source_url": donor["url"],
                        "perturbation": "source_swap",
                        "original_source_id": group["source_id"],
                        "original_source_url": group["url"],
                        "donor_source_id": donor["source_id"],
                        "source_swap_identity_contract": "balanced_direct_publisher_v1",
                    },
                }
            )
            negative = CandidateRoute(
                evidence=negative_evidence,
                gold_route=False,
                support_label=SupportLabel.SUPPORTED,
                source_match=False,
                temporal_valid=True,
                version_valid=True,
                completeness=1.0,
                risk_type=RiskType.WRONG_ATTRIBUTION,
            )
            candidates = [positive, negative]
            candidates.sort(key=stable_candidate_order)
            claim = ClaimRecord(
                claim_id=claim_id,
                query_id=query_id,
                text=group["text"],
                claim_type="finfact_cited_evidence_statement",
                temporal_scope=TemporalScope.UNSPECIFIED,
                metadata={"parent_claim": item["claim"], "evidence_index": group["evidence_index"]},
            )
            route_groups.append(
                ClaimRouteGroup(
                    group_id=stable_id("finfact-source-group-v1", claim_id),
                    query=query,
                    claim=claim,
                    candidates=tuple(candidates),
                    split="cross_dataset_final_holdout",
                    dataset="finfact_publisher_source_swap_v1",
                )
            )
        bundles.append(
            ClaimBundle(
                bundle_id=stable_id("finfact-source-bundle-v1", query_id),
                query=query,
                groups=tuple(route_groups),
            )
        )
    validate_dataset(bundles)
    positive_sources = Counter(
        candidate.evidence.source_id
        for bundle in bundles
        for group in bundle.groups
        for candidate in group.candidates
        if candidate.gold_route
    )
    negative_sources = Counter(
        candidate.evidence.source_id
        for bundle in bundles
        for group in bundle.groups
        for candidate in group.candidates
        if not candidate.gold_route
    )
    if positive_sources != negative_sources:
        raise RuntimeError("Fin-Fact positive and negative publisher histograms differ")
    if any(
        len({c.evidence.source_id for c in group.candidates}) != 2
        for bundle in bundles
        for group in bundle.groups
    ):
        raise RuntimeError("Fin-Fact contains a same-publisher source swap")
    summary = {
        **dict(sorted(audit.items())),
        "positive_negative_source_histograms_equal": True,
        "all_donor_candidates_used_once": len(set(matching.values())) == len(flat),
        "unique_publishers": len(positive_sources),
        "largest_publisher_count": max(positive_sources.values()),
        "largest_publisher_rate": max(positive_sources.values()) / len(flat),
    }
    return bundles, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protected", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("Fin-Fact input must be a JSON list")
    protected = protected_signatures(args.protected)
    bundles, audit = build_bundles(rows, protected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_bundles_jsonl(args.output, bundles)
    report = {
        "status": "frozen_before_model_evaluation",
        "version": "finfact_publisher_source_swap_v1_balanced_direct_publishers",
        "input": str(args.input),
        "protected": [str(path) for path in args.protected],
        "output": str(args.output),
        "counts": {
            "input_claims": len(rows),
            "output_bundles": len(bundles),
            "output_groups": sum(len(bundle.groups) for bundle in bundles),
            "output_candidates": sum(
                len(group.candidates) for bundle in bundles for group in bundle.groups
            ),
        },
        "audit": audit,
        "protected_signature_counts": {name: len(values) for name, values in protected.items()},
        "excluded_hosts": sorted(EXCLUDED_HOSTS),
        "sha256": {
            "input": sha256_file(args.input),
            "output": sha256_file(args.output),
            **{f"protected_{i}": sha256_file(path) for i, path in enumerate(args.protected)},
        },
        "policy": [
            "Final untouched financial-domain publisher-attribution holdout; freeze before scoring.",
            "Images are never loaded; only claim, cited evidence sentence, and direct publisher URL are visible.",
            "Archive hosts, shorteners, fact-check self citations, invalid links, and multi-publisher sentences are excluded.",
            "Exact normalized claim, canonical URL, or evidence-text overlap with any protected split removes the whole claim.",
            "Every positive publisher is used exactly once as a legal negative donor.",
            "Positive and negative publisher histograms are identical and candidate order is label-independent.",
        ],
    }
    args.manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
