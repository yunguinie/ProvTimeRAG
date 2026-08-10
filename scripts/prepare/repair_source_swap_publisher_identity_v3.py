"""Replace archive access hosts with recoverable publisher identities.

Wayback URLs are access locations, not publisher identities. This migration
extracts the embedded publisher URL, preserves the archive URL as provenance,
updates source/document identities, and removes Source-Swap groups whose gold
and negative publisher identities collapse after canonicalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from provtimerag.data import read_bundles_jsonl, validate_dataset, write_bundles_jsonl
from provtimerag.data.adapters.common import stable_id
from scripts.evaluate.audit_archive_source_identity_v2 import wayback_target_url


ARCHIVE_HOSTS = {"web.archive.org", "www.web.archive.org"}


def url_host(value: Any) -> str:
    return (urlparse(str(value or "").strip()).hostname or "").lower()


def recover_target(metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    for field in ("source_url", "cached_source_url", "swapped_source_url"):
        target = wayback_target_url(metadata.get(field))
        if target and url_host(target) not in ARCHIVE_HOSTS:
            return target, field
    return None, None


def canonicalize_evidence(evidence: Any) -> tuple[Any, dict[str, Any]]:
    legacy_source = evidence.source_id.casefold()
    audit = {
        "archive_candidate": legacy_source in ARCHIVE_HOSTS,
        "resolved": False,
        "legacy_source_id": evidence.source_id,
        "publisher_source_id": evidence.source_id,
    }
    if legacy_source not in ARCHIVE_HOSTS:
        return evidence, audit
    metadata = dict(evidence.metadata)
    target_url, resolution_field = recover_target(metadata)
    if not target_url:
        metadata.update(
            {
                "publisher_identity_contract": "archive_unresolved_v3",
                "publisher_identity_method": "unresolved_wayback_target",
                "access_source_id": evidence.source_id,
            }
        )
        return evidence.model_copy(update={"metadata": metadata}), audit

    publisher = url_host(target_url)
    access_url = metadata.get(resolution_field) if resolution_field else None
    legacy_source_url = metadata.get("source_url")
    legacy_cached_source_url = metadata.get("cached_source_url")
    metadata.update(
        {
            "publisher_identity_contract": "publisher_not_archive_access_host_v3",
            "publisher_identity_method": "wayback_embedded_target_url",
            "publisher_source_id": publisher,
            "publisher_source_url": target_url,
            "access_source_id": evidence.source_id,
            "archive_access_url": access_url,
            "legacy_source_url": legacy_source_url,
            "legacy_cached_source_url": legacy_cached_source_url,
            "source_url": target_url,
            "cached_source_url": access_url,
        }
    )
    if metadata.get("swapped_source_url") and wayback_target_url(
        metadata.get("swapped_source_url")
    ):
        metadata["legacy_swapped_source_url"] = metadata["swapped_source_url"]
        metadata["swapped_source_url"] = target_url
    repaired = evidence.model_copy(
        update={
            "source_id": publisher,
            "document_id": stable_id("publisher-document-v3", target_url),
            "metadata": metadata,
        }
    )
    audit.update(
        {
            "resolved": True,
            "publisher_source_id": publisher,
            "resolution_field": resolution_field,
        }
    )
    return repaired, audit


def repair_bundle(bundle: Any) -> tuple[Any | None, Counter[str]]:
    counts: Counter[str] = Counter()
    repaired_groups = []
    for group in bundle.groups:
        old_candidates = list(group.candidates)
        new_candidates = []
        candidate_audits = []
        for candidate in old_candidates:
            evidence, audit = canonicalize_evidence(candidate.evidence)
            candidate_audits.append(audit)
            new_candidates.append(candidate.model_copy(update={"evidence": evidence}))
            counts["candidates"] += 1
            counts["archive_candidates"] += int(audit["archive_candidate"])
            counts["resolved_archive_candidates"] += int(audit["resolved"])
            counts["changed_source_ids"] += int(
                audit["legacy_source_id"] != audit["publisher_source_id"]
            )

        gold_indices = [
            index for index, candidate in enumerate(old_candidates) if candidate.gold_route
        ]
        for index, (old_candidate, new_candidate) in enumerate(
            zip(old_candidates, new_candidates, strict=True)
        ):
            metadata = dict(new_candidate.evidence.metadata)
            if (
                not old_candidate.gold_route
                and str(metadata.get("perturbation", "")).casefold() == "source_swap"
            ):
                counts["source_swap_negatives"] += 1
                legacy_original = str(metadata.get("original_source_id") or "")
                matches = [
                    gold_index
                    for gold_index in gold_indices
                    if old_candidates[gold_index].evidence.source_id == legacy_original
                ]
                anchor_index = matches[0] if matches else gold_indices[0]
                canonical_original = new_candidates[anchor_index].evidence.source_id
                metadata["legacy_original_source_id"] = legacy_original
                metadata["original_source_id"] = canonical_original
                metadata["source_swap_identity_contract"] = (
                    "publisher_source_id_v3"
                )
                new_candidates[index] = new_candidate.model_copy(
                    update={
                        "evidence": new_candidate.evidence.model_copy(
                            update={"metadata": metadata}
                        )
                    }
                )

        gold_sources = {
            candidate.evidence.source_id
            for candidate in new_candidates
            if candidate.gold_route
        }
        negative_sources = {
            candidate.evidence.source_id
            for candidate in new_candidates
            if not candidate.gold_route
            and str(candidate.evidence.metadata.get("perturbation", "")).casefold()
            == "source_swap"
        }
        if gold_sources & negative_sources:
            counts["publisher_identity_collisions"] += 1
            counts["dropped_groups"] += 1
            continue
        repaired_groups.append(group.model_copy(update={"candidates": tuple(new_candidates)}))
        counts["groups"] += 1

    if not repaired_groups:
        counts["dropped_bundles"] += 1
        return None, counts
    return bundle.model_copy(update={"groups": tuple(repaired_groups)}), counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    repaired = []
    totals: Counter[str] = Counter()
    input_bundles = 0
    for bundle in read_bundles_jsonl(args.input):
        input_bundles += 1
        item, counts = repair_bundle(bundle)
        totals.update(counts)
        if item is not None:
            repaired.append(item)
    if not repaired:
        raise RuntimeError("publisher identity migration produced no bundles")
    validate_dataset(repaired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_bundles_jsonl(args.output, repaired)
    manifest = {
        "status": "complete",
        "version": "source_swap_publisher_identity_v3",
        "input": str(args.input),
        "output": str(args.output),
        "input_bundles": input_bundles,
        "output_bundles": len(repaired),
        "audit": dict(sorted(totals.items())),
        "sha256": {
            "input": sha256_file(args.input),
            "output": sha256_file(args.output),
        },
        "policy": [
            "Wayback host is retained as archive_access_url, not publisher identity.",
            "Recoverable embedded target URL defines source_id and source_url.",
            "Labels and candidate order are unchanged unless canonicalization makes a Source-Swap negative identical to a gold publisher.",
            "Publisher-identity collisions are removed and counted.",
        ],
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
