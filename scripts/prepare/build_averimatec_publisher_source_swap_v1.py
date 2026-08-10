"""Build a frozen cross-dataset publisher Source-Swap benchmark from AVerImaTeC.

The benchmark uses claim/question/annotated-answer text and publisher URLs only;
images are never loaded.  Archive access URLs are unwrapped when possible and
otherwise excluded.  Every answerable question receives one wrong-publisher
negative whose evidence text is unchanged, and candidate order is determined
without consulting labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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
    read_bundles_jsonl,
    validate_dataset,
    write_bundles_jsonl,
)
from provtimerag.data.adapters.common import stable_id
from scripts.evaluate.audit_archive_source_identity_v2 import wayback_target_url


UNRESOLVED_ARCHIVE_HOSTS = {
    "archive.is",
    "archive.ph",
    "archive.vn",
    "ghostarchive.org",
    "perma.cc",
    "web.archive.org",
    "www.web.archive.org",
}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def text_fingerprint(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def canonical_publisher_url(value: Any) -> tuple[str | None, str | None, str]:
    access_url = str(value or "").strip()
    if not access_url:
        return None, None, "missing_url"
    target = wayback_target_url(access_url)
    candidate = target or access_url
    parsed = urlsplit(candidate)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme.casefold() not in {"http", "https"} or not host:
        return None, None, "invalid_url"
    if host in UNRESOLVED_ARCHIVE_HOSTS:
        return None, None, "unresolved_archive"
    port = f":{parsed.port}" if parsed.port else ""
    canonical = urlunsplit(
        (parsed.scheme.casefold(), f"{host}{port}", parsed.path or "/", parsed.query, "")
    )
    return canonical, host, "wayback_unwrapped" if target else "direct_publisher"


def protected_signatures(paths: list[Path]) -> dict[str, set[str]]:
    signatures = {"queries": set(), "urls": set(), "texts": set()}
    for path in paths:
        for bundle in read_bundles_jsonl(path):
            signatures["queries"].add(normalize_text(bundle.query.text))
            for group in bundle.groups:
                signatures["queries"].add(normalize_text(group.claim.text))
                for candidate in group.candidates:
                    signatures["texts"].add(text_fingerprint(candidate.evidence.text))
                    url, _, _ = canonical_publisher_url(
                        candidate.evidence.metadata.get("source_url")
                    )
                    if url:
                        signatures["urls"].add(url)
    return signatures


def parse_groups(record: dict[str, Any], row_index: int, audit: Counter[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for question_index, question in enumerate(record.get("questions") or []):
        question_text = " ".join(str(question.get("question") or "").split())
        if not question_text:
            audit["questions_missing_text"] += 1
            continue
        positives = []
        for answer_index, answer in enumerate(question.get("answers") or []):
            answer_text = " ".join(str(answer.get("answer_text") or "").split())
            url, source_id, method = canonical_publisher_url(answer.get("source_url"))
            if not answer_text:
                audit["answers_missing_text"] += 1
                continue
            if not url or not source_id:
                audit[f"answers_excluded_{method}"] += 1
                continue
            audit[f"publisher_url_{method}"] += 1
            positives.append(
                {
                    "answer_index": answer_index,
                    "text": answer_text,
                    "url": url,
                    "source_id": source_id,
                    "access_url": str(answer.get("source_url") or "").strip(),
                    "answer_type": answer.get("answer_type"),
                    "source_medium": answer.get("source_medium"),
                }
            )
        if positives:
            groups.append(
                {
                    "row_index": row_index,
                    "question_index": question_index,
                    "question": question_text,
                    "question_type": question.get("question_type") or [],
                    "answer_method": question.get("answer_method"),
                    "input_images": question.get("input_images") or [],
                    "positives": positives,
                }
            )
        else:
            audit["questions_without_usable_answers"] += 1
    return groups


def group_signatures(claim_text: str, groups: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "queries": {normalize_text(claim_text)} | {normalize_text(g["question"]) for g in groups},
        "urls": {a["url"] for g in groups for a in g["positives"]},
        "texts": {text_fingerprint(a["text"]) for g in groups for a in g["positives"]},
    }


def stable_candidate_order(candidate: CandidateRoute) -> str:
    evidence = candidate.evidence
    value = f"{normalize_text(evidence.text)}\0{evidence.metadata.get('source_url', '')}"
    return hashlib.sha256(f"averimatec-candidate-order-v1\0{value}".encode()).hexdigest()


def build_bundles(
    rows: list[dict[str, Any]], protected: dict[str, set[str]]
) -> tuple[list[ClaimBundle], Counter[str]]:
    audit: Counter[str] = Counter()
    prepared: list[dict[str, Any]] = []
    for row_index, record in enumerate(rows):
        claim_text = " ".join(str(record.get("claim_text") or "").split())
        if not claim_text:
            audit["claims_missing_text"] += 1
            continue
        groups = parse_groups(record, row_index, audit)
        if len(groups) < 2:
            audit["claims_excluded_fewer_than_two_groups"] += 1
            continue
        signatures = group_signatures(claim_text, groups)
        overlap = {name: signatures[name] & protected[name] for name in signatures}
        if any(overlap.values()):
            audit["claims_excluded_protected_overlap"] += 1
            for name, values in overlap.items():
                audit[f"protected_{name}_overlap_items"] += len(values)
            continue
        prepared.append(
            {"row_index": row_index, "record": record, "claim_text": claim_text, "groups": groups}
        )

    donor_pool = [
        (item["row_index"], group["question_index"], answer)
        for item in prepared
        for group in item["groups"]
        for answer in group["positives"]
    ]
    donor_pool.sort(
        key=lambda item: hashlib.sha256(
            f"averimatec-donor-v1\0{item[0]}\0{item[1]}\0{item[2]['url']}".encode()
        ).hexdigest()
    )
    bundles: list[ClaimBundle] = []
    for item in prepared:
        row_index = item["row_index"]
        record = item["record"]
        query_id = stable_id("averimatec-query-v1", f"val:{row_index}:{item['claim_text']}")
        query = QueryRecord(
            query_id=query_id,
            text=item["claim_text"],
            metadata={
                "dataset": "averimatec",
                "label": record.get("label"),
                "date": record.get("date"),
                "text_evidence_only": True,
                "images_loaded": False,
            },
        )
        route_groups = []
        for group in item["groups"]:
            claim_id = stable_id("averimatec-subclaim-v1", f"{query_id}:{group['question_index']}")
            candidates: list[CandidateRoute] = []
            for answer in group["positives"]:
                evidence_id = stable_id(
                    "averimatec-positive-v1",
                    f"{claim_id}:{answer['answer_index']}:{answer['url']}:{answer['text']}",
                )
                evidence = EvidenceRecord(
                    evidence_id=evidence_id,
                    text=answer["text"],
                    source_id=answer["source_id"],
                    source_role="publisher_web_evidence",
                    document_id=stable_id("averimatec-publisher-document-v1", answer["url"]),
                    version_id=stable_id("averimatec-unversioned-v1", answer["url"]),
                    metadata={
                        "source_url": answer["url"],
                        "archive_access_url": answer["access_url"] if answer["access_url"] != answer["url"] else None,
                        "source_medium": answer["source_medium"],
                        "answer_type": answer["answer_type"],
                        "publisher_identity_contract": "publisher_not_archive_access_host_v1",
                    },
                )
                candidates.append(
                    CandidateRoute(
                        evidence=evidence,
                        gold_route=True,
                        support_label=SupportLabel.SUPPORTED,
                        source_match=True,
                        temporal_valid=True,
                        version_valid=True,
                        completeness=1.0,
                        risk_type=RiskType.NONE,
                    )
                )
            gold_sources = {candidate.evidence.source_id for candidate in candidates}
            donor = next(
                (
                    answer
                    for donor_row, donor_question, answer in donor_pool
                    if (donor_row, donor_question) != (row_index, group["question_index"])
                    and answer["source_id"] not in gold_sources
                ),
                None,
            )
            if donor is None:
                audit["groups_without_distinct_donor"] += 1
                continue
            anchor = candidates[0].evidence
            negative_evidence = anchor.model_copy(
                update={
                    "evidence_id": stable_id(
                        "averimatec-source-swap-v1", f"{claim_id}:{donor['url']}:{anchor.text}"
                    ),
                    "source_id": donor["source_id"],
                    "document_id": stable_id("averimatec-publisher-document-v1", donor["url"]),
                    "version_id": stable_id("averimatec-unversioned-v1", donor["url"]),
                    "metadata": {
                        **anchor.metadata,
                        "source_url": donor["url"],
                        "archive_access_url": donor["access_url"] if donor["access_url"] != donor["url"] else None,
                        "perturbation": "source_swap",
                        "original_source_id": anchor.source_id,
                        "original_source_url": anchor.metadata.get("source_url"),
                        "donor_source_id": donor["source_id"],
                        "source_swap_identity_contract": "publisher_source_id_v1",
                    },
                }
            )
            candidates.append(
                CandidateRoute(
                    evidence=negative_evidence,
                    gold_route=False,
                    support_label=SupportLabel.SUPPORTED,
                    source_match=False,
                    temporal_valid=True,
                    version_valid=True,
                    completeness=1.0,
                    risk_type=RiskType.WRONG_ATTRIBUTION,
                )
            )
            candidates.sort(key=stable_candidate_order)
            claim = ClaimRecord(
                claim_id=claim_id,
                query_id=query_id,
                text=group["question"],
                claim_type="averimatec_text_evidence_question",
                temporal_scope=TemporalScope.UNSPECIFIED,
                metadata={
                    "parent_claim": item["claim_text"],
                    "question_type": group["question_type"],
                    "answer_method": group["answer_method"],
                    "input_image_count": len(group["input_images"]),
                    "images_loaded": False,
                },
            )
            route_groups.append(
                ClaimRouteGroup(
                    group_id=stable_id("averimatec-source-group-v1", claim_id),
                    query=query,
                    claim=claim,
                    candidates=tuple(candidates),
                    split="cross_dataset_holdout",
                    dataset="averimatec_publisher_source_swap_v1",
                )
            )
        if len(route_groups) < 2:
            audit["claims_excluded_after_donor_assignment"] += 1
            continue
        bundles.append(
            ClaimBundle(
                bundle_id=stable_id("averimatec-source-bundle-v1", query_id),
                query=query,
                groups=tuple(route_groups),
            )
        )
    return bundles, audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--protected", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("AVerImaTeC input must be a JSON list")
    protected = protected_signatures(args.protected)
    bundles, audit = build_bundles(rows, protected)
    if not bundles:
        raise RuntimeError("no eligible AVerImaTeC bundles remain")
    validate_dataset(bundles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_bundles_jsonl(args.output, bundles)
    sources = {
        candidate.evidence.source_id
        for bundle in bundles
        for group in bundle.groups
        for candidate in group.candidates
    }
    manifest = {
        "status": "frozen_before_model_evaluation",
        "version": "averimatec_publisher_source_swap_v1",
        "input": str(args.input),
        "protected": [str(path) for path in args.protected],
        "output": str(args.output),
        "counts": {
            "input_claims": len(rows),
            "output_bundles": len(bundles),
            "output_groups": sum(len(bundle.groups) for bundle in bundles),
            "output_candidates": sum(len(group.candidates) for bundle in bundles for group in bundle.groups),
            "publisher_sources": len(sources),
        },
        "audit": dict(sorted(audit.items())),
        "protected_signature_counts": {name: len(values) for name, values in protected.items()},
        "sha256": {
            "input": sha256_file(args.input),
            "output": sha256_file(args.output),
            **{f"protected_{index}": sha256_file(path) for index, path in enumerate(args.protected)},
        },
        "policy": [
            "Cross-dataset text-evidence publisher attribution stress test; images are never loaded.",
            "Questions may originate from image search, but model-visible evidence is annotated answer text plus publisher URL only.",
            "Wayback access URLs are unwrapped; unresolved archive access hosts are excluded.",
            "Candidate ordering is a stable hash independent of gold labels.",
            "Exact normalized claim/question text, canonical URL, or evidence-text overlap with protected splits removes the entire claim bundle.",
            "This manifest and output must be frozen before any checkpoint is evaluated.",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
