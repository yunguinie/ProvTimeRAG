"""Repair AVerImaTeC Source-Swap donors with deterministic perfect matching.

Version 1 selected the first legal donor from a globally sorted pool, causing
nearly every negative to expose the same publisher.  This migration assigns
each group's positive candidate to exactly one other group whenever a perfect
matching exists.  Consequently positive and negative publisher histograms are
identical, while same-group and same-publisher swaps remain forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from provtimerag.data import (
    ClaimBundle,
    read_bundles_jsonl,
    validate_dataset,
    write_bundles_jsonl,
)
from provtimerag.data.adapters.common import stable_id
from scripts.prepare.build_averimatec_publisher_source_swap_v1 import (
    sha256_file,
    stable_candidate_order,
)


def stable_hash(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def flatten_groups(bundles: list[ClaimBundle]) -> list[tuple[int, int, Any]]:
    return [
        (bundle_index, group_index, group)
        for bundle_index, bundle in enumerate(bundles)
        for group_index, group in enumerate(bundle.groups)
    ]


def positive_candidate(group: Any) -> Any:
    positives = [candidate for candidate in group.candidates if candidate.gold_route]
    if len(positives) != 1:
        raise ValueError(
            f"v2 balance repair requires exactly one gold candidate in {group.group_id}; "
            f"found {len(positives)}"
        )
    return positives[0]


def negative_candidate(group: Any) -> Any:
    negatives = [candidate for candidate in group.candidates if not candidate.gold_route]
    if len(negatives) != 1:
        raise ValueError(
            f"v2 balance repair requires exactly one negative candidate in {group.group_id}; "
            f"found {len(negatives)}"
        )
    return negatives[0]


def perfect_donor_matching(groups: list[Any]) -> dict[int, int]:
    """Return left-group to right-donor indices via deterministic Kuhn matching."""

    positives = [positive_candidate(group) for group in groups]
    adjacency: list[list[int]] = []
    for left, group in enumerate(groups):
        gold_sources = {
            candidate.evidence.source_id
            for candidate in group.candidates
            if candidate.gold_route
        }
        eligible = [
            right
            for right, donor in enumerate(positives)
            if right != left and donor.evidence.source_id not in gold_sources
        ]
        eligible.sort(
            key=lambda right: stable_hash(
                "averimatec-balanced-donor-v2",
                group.group_id,
                positives[right].evidence.evidence_id,
            )
        )
        if not eligible:
            raise RuntimeError(f"no legal donor for {group.group_id}")
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

    left_order = sorted(
        range(len(groups)),
        key=lambda left: stable_hash(
            "averimatec-balanced-left-v2", groups[left].group_id
        ),
    )
    for left in left_order:
        if not augment(left, set()):
            raise RuntimeError(
                f"no perfect legal donor matching; failed at {groups[left].group_id}"
            )
    if len(right_to_left) != len(groups):
        raise RuntimeError("donor matching is incomplete")
    return {left: right for right, left in right_to_left.items()}


def repair_bundles(bundles: list[ClaimBundle]) -> tuple[list[ClaimBundle], dict[str, Any]]:
    locations = flatten_groups(bundles)
    groups = [group for _, _, group in locations]
    positives = [positive_candidate(group) for group in groups]
    matching = perfect_donor_matching(groups)
    repaired_by_location: dict[tuple[int, int], Any] = {}
    for left, (bundle_index, group_index, group) in enumerate(locations):
        positive = positives[left]
        old_negative = negative_candidate(group)
        donor = positives[matching[left]]
        metadata = {
            **old_negative.evidence.metadata,
            "source_url": donor.evidence.metadata.get("source_url"),
            "archive_access_url": donor.evidence.metadata.get("archive_access_url"),
            "perturbation": "source_swap",
            "original_source_id": positive.evidence.source_id,
            "original_source_url": positive.evidence.metadata.get("source_url"),
            "donor_source_id": donor.evidence.source_id,
            "donor_evidence_id": donor.evidence.evidence_id,
            "source_swap_identity_contract": "balanced_publisher_perfect_matching_v2",
            "legacy_v1_negative_source_id": old_negative.evidence.source_id,
        }
        evidence = old_negative.evidence.model_copy(
            update={
                "evidence_id": stable_id(
                    "averimatec-source-swap-v2",
                    f"{group.group_id}:{donor.evidence.evidence_id}:{positive.evidence.text}",
                ),
                "text": positive.evidence.text,
                "source_id": donor.evidence.source_id,
                "document_id": donor.evidence.document_id,
                "version_id": donor.evidence.version_id,
                "metadata": metadata,
            }
        )
        negative = old_negative.model_copy(update={"evidence": evidence})
        candidates = [positive, negative]
        candidates.sort(key=stable_candidate_order)
        repaired_by_location[(bundle_index, group_index)] = group.model_copy(
            update={
                "candidates": tuple(candidates),
                "dataset": "averimatec_publisher_source_swap_v2",
            }
        )

    repaired = []
    for bundle_index, bundle in enumerate(bundles):
        groups_for_bundle = tuple(
            repaired_by_location[(bundle_index, group_index)]
            for group_index in range(len(bundle.groups))
        )
        repaired.append(bundle.model_copy(update={"groups": groups_for_bundle}))
    validate_dataset(repaired)

    positive_sources = Counter(
        candidate.evidence.source_id
        for bundle in repaired
        for group in bundle.groups
        for candidate in group.candidates
        if candidate.gold_route
    )
    negative_sources = Counter(
        candidate.evidence.source_id
        for bundle in repaired
        for group in bundle.groups
        for candidate in group.candidates
        if not candidate.gold_route
    )
    invalid_same_source = sum(
        bool(
            {c.evidence.source_id for c in group.candidates if c.gold_route}
            & {c.evidence.source_id for c in group.candidates if not c.gold_route}
        )
        for bundle in repaired
        for group in bundle.groups
    )
    invalid_text_changes = sum(
        negative_candidate(group).evidence.text != positive_candidate(group).evidence.text
        for bundle in repaired
        for group in bundle.groups
    )
    if positive_sources != negative_sources:
        raise RuntimeError("positive and negative publisher histograms differ")
    if invalid_same_source or invalid_text_changes:
        raise RuntimeError(
            f"invalid repaired swaps: same_source={invalid_same_source}, "
            f"text_changes={invalid_text_changes}"
        )
    audit = {
        "groups": len(groups),
        "unique_positive_sources": len(positive_sources),
        "unique_negative_sources": len(negative_sources),
        "positive_negative_source_histograms_equal": True,
        "largest_source_count": max(positive_sources.values()),
        "largest_source_rate": max(positive_sources.values()) / len(groups),
        "invalid_same_source_groups": invalid_same_source,
        "invalid_text_change_groups": invalid_text_changes,
        "all_donor_candidates_used_once": len(set(matching.values())) == len(groups),
    }
    return repaired, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    bundles = list(read_bundles_jsonl(args.input))
    repaired, audit = repair_bundles(bundles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_bundles_jsonl(args.output, repaired)
    v1_manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    report = {
        "status": "frozen_before_model_evaluation",
        "version": "averimatec_publisher_source_swap_v2_balanced_donors",
        "input": str(args.input),
        "input_manifest": str(args.input_manifest),
        "output": str(args.output),
        "counts": {
            "bundles": len(repaired),
            "groups": sum(len(bundle.groups) for bundle in repaired),
            "candidates": sum(
                len(group.candidates) for bundle in repaired for group in bundle.groups
            ),
        },
        "audit": audit,
        "inherited_protected_inputs": v1_manifest.get("protected"),
        "inherited_protected_sha256": {
            key: value
            for key, value in v1_manifest.get("sha256", {}).items()
            if key.startswith("protected_")
        },
        "sha256": {
            "input": sha256_file(args.input),
            "input_manifest": sha256_file(args.input_manifest),
            "output": sha256_file(args.output),
        },
        "supersedes": {
            "version": "averimatec_publisher_source_swap_v1",
            "reason": "v1 first-legal-donor selection collapsed 163/164 negatives onto www.wsj.com",
            "v1_model_results_valid": False,
        },
        "policy": [
            "The v1 claim/group eligibility and leakage exclusions are unchanged.",
            "Each positive candidate is used exactly once as a donor through deterministic perfect matching.",
            "Positive and negative publisher histograms must be identical.",
            "Same-group and same-publisher donor assignments are forbidden.",
            "Evidence text is unchanged by Source-Swap and candidate ordering remains label-independent.",
            "Freeze this output before any v2 checkpoint evaluation.",
        ],
    }
    args.manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
