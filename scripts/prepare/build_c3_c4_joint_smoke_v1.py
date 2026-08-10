"""Build paired C3+C4 generation queues from an untouched multi-claim Source Swap slice.

The output contains the same claim groups under three conditions:
raw all candidates, frozen C2 independent routing, and C3 structured assignment.
No API call is made here; generation is a separate frozen step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from scripts.evaluate import run_c3_source_swap_v7 as c3
from scripts.evaluate import run_c3_structured_source_swap_coverage as c3_structured
from scripts.prepare.build_c4_smoke import prompt_for
from provtimerag.routing.multitask_model import MultiTaskProvenanceRouter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_answers(path: Path) -> dict[tuple[str, str], str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    answers: dict[tuple[str, str], str] = {}
    for record in records:
        claim_text = str(record.get("claim") or "").strip()
        for question in record.get("questions", []):
            values = question.get("answers") or []
            if not values:
                continue
            text = str(question.get("question") or "").strip()
            answer = str(values[0].get("answer") or "").strip()
            key = (claim_text, text)
            if claim_text and text and answer:
                previous = answers.get(key)
                if previous is not None and previous != answer:
                    raise ValueError(
                        f"conflicting answers for claim/question: {claim_text} / {text}"
                    )
                answers[key] = answer
    return answers


def evidence_json(item: Any) -> dict[str, Any]:
    return item.evidence.model_dump(mode="json")


def risk_values(group: Any) -> list[str]:
    values: set[str] = set()
    for candidate in group.candidates:
        value = candidate.risk_type
        values.add(str(getattr(value, "value", value)))
    return sorted(values)


def make_rows(
    bundles: list[Any],
    scores: dict[str, list[Any]],
    answers: dict[tuple[str, str], str],
    chooser: Callable[[Any, dict[str, list[Any]]], tuple[int, ...]] | None,
    *,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle_index, bundle in enumerate(bundles, start=1):
        if chooser is None:
            choices = tuple(
                None for _ in bundle.groups
            )
        else:
            choices = chooser(bundle, scores)
        for group_index, (group, choice) in enumerate(
            zip(bundle.groups, choices, strict=True), start=1
        ):
            question = group.claim.text.strip()
            metadata = group.claim.metadata or {}
            parent_claim = str(metadata.get("parent_claim") or "").strip()
            answer_key = (parent_claim, question)
            if answer_key not in answers:
                raise ValueError(
                    "missing AVeriTeC answer for claim/question: "
                    f"{parent_claim} / {question}"
                )
            all_candidates = [evidence_json(candidate) for candidate in group.candidates]
            if choice is None:
                selected_indices: list[int] = []
                selected_candidates = all_candidates
            else:
                selected_indices = [int(choice)]
                selected_candidates = [all_candidates[int(choice)]]
            request_id = f"c3c4-joint-{bundle_index:04d}-{group_index:02d}"
            rows.append(
                {
                    "request_id": request_id,
                    "split": "c3_c4_joint_smoke",
                    "method": method,
                    "bundle_id": bundle.bundle_id,
                    "query_id": bundle.query.query_id,
                    "claim_id": group.claim.claim_id,
                    "group_id": group.group_id,
                    "prompt": prompt_for(
                        group.claim.model_dump(mode="json"),
                        [{"evidence": item} for item in selected_candidates],
                    ),
                    "candidates": selected_candidates,
                    "gold": {
                        "answer": answers[answer_key],
                        "gold_evidence_ids": [
                            candidate.evidence.evidence_id
                            for candidate in group.candidates
                            if candidate.gold_route
                        ],
                        "risk_types": risk_values(group),
                    },
                    "routing": {
                        "method": method,
                        "selected_indices": selected_indices,
                        "selected_evidence_ids": [
                            selected_candidates[index]["evidence_id"]
                            for index in range(len(selected_candidates))
                        ],
                        "source_group_id": group.group_id,
                    },
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-input",
        type=Path,
        default=Path("data/processed/router_v1/train/source_swap.jsonl"),
    )
    parser.add_argument(
        "--dev-input",
        type=Path,
        default=Path("data/processed/c3_blind_v1/source_swap.jsonl"),
    )
    parser.add_argument(
        "--answers-input",
        type=Path,
        default=Path("data/raw/averitec/dev.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit-bundles", type=int, default=100)
    parser.add_argument("--dev-limit-bundles", type=int, default=20)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--group-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration fraction must be between zero and one")

    c3._patch_feature_functions()
    train_bundles = c3.base.load_bundles(args.train_input, args.train_limit_bundles)
    dev_bundles = c3.base.load_bundles(args.dev_input, args.dev_limit_bundles)
    if not train_bundles or not dev_bundles:
        raise ValueError("train and dev bundles must be non-empty")
    calibration_size = max(1, int(round(len(train_bundles) * args.calibration_fraction)))
    if calibration_size >= len(train_bundles):
        raise ValueError("training limit is too small for the calibration fraction")
    fit_bundles = train_bundles[:-calibration_size]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint / "tokenizer", local_files_only=True
    )
    model = MultiTaskProvenanceRouter.from_pretrained(
        args.checkpoint / "backbone", dropout=0.1
    )
    model.heads.load_state_dict(
        torch.load(args.checkpoint / "router_heads.pt", map_location="cpu", weights_only=True)
    )
    model.to(
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).eval()

    fit_scores = c3.base.route_scores(
        model,
        tokenizer,
        fit_bundles,
        batch_size=args.group_batch_size,
        max_length=args.max_length,
        device=device,
    )
    dev_scores = c3.base.route_scores(
        model,
        tokenizer,
        dev_bundles,
        batch_size=args.group_batch_size,
        max_length=args.max_length,
        device=device,
    )
    weight = c3_structured.implementation.fit_assignment_head(
        fit_bundles,
        fit_scores,
        args.candidate_limit,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    independent = lambda bundle, scores: c3_structured.implementation.choose_independent(
        bundle, scores
    )
    structured = lambda bundle, scores: c3_structured.implementation.choose_structured(
        bundle, scores, weight, args.candidate_limit
    )
    answers = load_answers(args.answers_input)

    raw_rows = make_rows(
        dev_bundles, dev_scores, answers, None, method="raw_all_candidates"
    )
    c2_rows = make_rows(
        dev_bundles, dev_scores, answers, independent, method="c2_temporal_independent"
    )
    c3_rows = make_rows(
        dev_bundles, dev_scores, answers, structured, method="c3_structured_assignment"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_queue.jsonl"
    c2_path = args.output_dir / "c2_queue.jsonl"
    c3_path = args.output_dir / "c3_queue.jsonl"
    write_jsonl(raw_path, raw_rows)
    write_jsonl(c2_path, c2_rows)
    write_jsonl(c3_path, c3_rows)
    manifest = {
        "status": "complete",
        "version": "c3_c4_joint_smoke_v1",
        "train_bundles": len(train_bundles),
        "fit_bundles": len(fit_bundles),
        "calibration_bundles": calibration_size,
        "dev_bundles": len(dev_bundles),
        "dev_groups": len(raw_rows),
        "methods": {
            "raw": str(raw_path),
            "c2": str(c2_path),
            "c3": str(c3_path),
        },
        "learned_weights": weight.tolist(),
        "sha256": {
            "raw": sha256_file(raw_path),
            "c2": sha256_file(c2_path),
            "c3": sha256_file(c3_path),
        },
        "policy": [
            "Development smoke only; do not tune after API outputs.",
            "All three methods use the same claim groups and gold answers.",
            "Raw, C2, and C3 generation must use the same frozen generator contract.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

