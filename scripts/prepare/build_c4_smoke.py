"""Build a non-leaking C4 generation smoke queue from HoH test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_for(claim: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    question = str(claim.get("metadata", {}).get("question") or claim.get("text", ""))
    evidence_lines = []
    for index, candidate in enumerate(candidates, start=1):
        evidence = candidate["evidence"]
        metadata = evidence.get("metadata", {})
        evidence_lines.append(
            "\n".join(
                [
                    f"Evidence E{index}",
                    f"source_id: {evidence.get('source_id')}",
                    f"document_title: {metadata.get('document_title', '')}",
                    f"publication_time: {evidence.get('publication_time')}",
                    f"valid_from: {evidence.get('valid_from')}",
                    f"valid_to: {evidence.get('valid_to')}",
                    f"supersedes: {', '.join(evidence.get('supersedes_ids', []))}",
                    f"text: {evidence.get('text', '')}",
                ]
            )
        )
    return (
        "You are a provenance-constrained answer generator.\n"
        "Answer the question only from the supplied evidence. Do not use outside knowledge.\n"        "For yes/no questions, begin the answer with exactly Yes or No; you may add a cited explanation after it.\n"
        "A citation must be one of E1, E2, ... and every factual sentence must cite at least one evidence ID.\n"
        "If the evidence is stale, contradictory, insufficient, or cannot support the claim, abstain or state the conflict.\n"
        "Return one JSON object only with this schema:\n"
        '{"answer":"string","citation_ids":["E1"],"abstain":false,'
        '"state":"SUPPORTED|STALE|WRONG_SOURCE|CONTRADICTED|PARTIALLY_SUPPORTED|INSUFFICIENT"}\n\n'
        f"Question: {question}\n\n"
        + "\n\n".join(evidence_lines)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/interim/hoh/test.jsonl"))
    parser.add_argument("--pilot", type=Path, default=Path("data/processed/pilot/pilot.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/c4_v1/smoke20.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/c4_v1/smoke20_manifest.json"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    test_rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    pilot_rows = [json.loads(line) for line in args.pilot.read_text(encoding="utf-8").splitlines() if line.strip()]
    pilot_queries = {row["query"]["query_id"] for row in pilot_rows}
    selected: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    for bundle in test_rows:
        if bundle["query"]["query_id"] in pilot_queries:
            continue
        group = bundle["groups"][0]
        if not any(candidate.get("risk_type") == "stale" for candidate in group["candidates"]):
            continue
        entity = str(group["claim"].get("entity_id") or "")
        if entity and entity in seen_entities:
            continue
        candidates = group["candidates"]
        claim = group["claim"]
        gold_ids = [candidate["evidence"]["evidence_id"] for candidate in candidates if candidate.get("gold_route")]
        gold_answer = claim.get("metadata", {}).get("gold_answer")
        if not gold_ids or not gold_answer:
            continue
        selected.append(
            {
                "request_id": f"c4-smoke-{len(selected)+1:03d}",
                "split": "c4_v1_smoke",
                "bundle_id": bundle["bundle_id"],
                "query_id": bundle["query"]["query_id"],
                "claim_id": claim["claim_id"],
                "claim_type": claim.get("claim_type"),
                "prompt": prompt_for(claim, candidates),
                "candidates": [candidate["evidence"] for candidate in candidates],
                "gold": {
                    "answer": gold_answer,
                    "gold_evidence_ids": gold_ids,
                    "risk_types": sorted({candidate["risk_type"] for candidate in candidates}),
                },
            }
        )
        if entity:
            seen_entities.add(entity)
        if len(selected) == args.limit:
            break
    if len(selected) != args.limit:
        raise RuntimeError(f"selected {len(selected)} records, expected {args.limit}")
    if any(row["gold"]["answer"] in row["prompt"] for row in selected):
        raise RuntimeError("gold answer leaked into a smoke prompt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    manifest = {
        "version": "c4_v1_smoke",
        "status": "development_only",
        "input": str(args.input),
        "pilot_exclusion": str(args.pilot),
        "records": len(selected),
        "risk_scope": "stale_only",
        "pilot_query_overlap": 0,
        "gold_answer_in_prompt": False,
        "sha256": {"input": sha256_file(args.input), "output": sha256_file(args.output)},
        "use_policy": [
            "Smoke only; do not use for final claims or prompt tuning after API output is observed.",
            "Use the same generator and contract for all compared routing methods.",
            "A full C4 evaluation must add untouched Source Swap and Insufficient slices.",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
