"""Build C3+C4 queues with a calibrated minimum-sufficient evidence guard.

The primary C2/C3 route decision remains unchanged.  On calibration bundles only,
we select a score-margin threshold and retain a second candidate for uncertain groups
so that single-candidate routing does not destroy evidence recall.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from transformers import AutoTokenizer

from provtimerag.routing.multitask_model import MultiTaskProvenanceRouter
from scripts.evaluate import run_c3_source_swap_v7 as c3
from scripts.evaluate import run_c3_structured_source_swap_coverage as c3_structured
from scripts.prepare import build_c3_c4_joint_smoke_v1 as v1
from scripts.prepare.build_c4_smoke import prompt_for


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_indices(group: Any) -> list[int]:
    """Stable candidate order independent of gold_route and routing scores."""
    return sorted(
        range(len(group.candidates)),
        key=lambda index: hashlib.sha256(
            f"{group.claim.claim_id}|{group.candidates[index].evidence.evidence_id}".encode("utf-8")
        ).hexdigest(),
    )


def primary_independent(bundle: Any, scores: dict[str, list[Any]]) -> tuple[int, ...]:
    return tuple(
        int(np.argmax([item.route_score for item in scores[group.group_id]]))
        for group in bundle.groups
    )


def primary_structured(
    bundle: Any,
    scores: dict[str, list[Any]],
    weight: np.ndarray,
    candidate_limit: int,
) -> tuple[int, ...]:
    return c3_structured.implementation.choose_structured(
        bundle, scores, weight, candidate_limit
    )


def expanded_choices(
    bundle: Any,
    scores: dict[str, list[Any]],
    primary: tuple[int, ...],
    margin_threshold: float,
) -> tuple[tuple[int, ...], ...]:
    output: list[tuple[int, ...]] = []
    for group, selected_index in zip(bundle.groups, primary, strict=True):
        values = scores[group.group_id]
        chosen = [int(selected_index)]
        if len(values) > 1:
            ranked = sorted(
                range(len(values)),
                key=lambda index: values[index].route_score,
                reverse=True,
            )
            alternate = ranked[0] if ranked[0] != selected_index else ranked[1]
            gap = float(values[selected_index].route_score - values[alternate].route_score)
            if gap < margin_threshold:
                chosen.append(int(alternate))
        output.append(tuple(dict.fromkeys(chosen)))
    return tuple(output)


def coverage_report(
    bundles: list[Any],
    scores: dict[str, list[Any]],
    primary_fn: Callable[[Any, dict[str, list[Any]]], tuple[int, ...]],
    margin_threshold: float,
) -> dict[str, float | int]:
    groups = covered = selected = 0
    for bundle in bundles:
        primary = primary_fn(bundle, scores)
        choices = expanded_choices(bundle, scores, primary, margin_threshold)
        for group, indices in zip(bundle.groups, choices, strict=True):
            groups += 1
            selected += len(indices)
            gold = {
                candidate.evidence.evidence_id
                for candidate in group.candidates
                if candidate.gold_route
            }
            chosen = {scores[group.group_id][index].evidence.evidence_id for index in indices}
            covered += int(bool(gold & chosen))
    return {
        "groups": groups,
        "gold_coverage": covered / groups if groups else 0.0,
        "covered_groups": covered,
        "mean_selected_candidates": selected / groups if groups else 0.0,
    }


def calibrate_margin(
    bundles: list[Any],
    scores: dict[str, list[Any]],
    primary_fn: Callable[[Any, dict[str, list[Any]]], tuple[int, ...]],
    target_coverage: float,
) -> dict[str, Any]:
    margins: list[float] = []
    for bundle in bundles:
        primary = primary_fn(bundle, scores)
        for group, selected_index in zip(bundle.groups, primary, strict=True):
            values = scores[group.group_id]
            if len(values) < 2:
                continue
            ranked = sorted(
                range(len(values)),
                key=lambda index: values[index].route_score,
                reverse=True,
            )
            alternate = ranked[0] if ranked[0] != selected_index else ranked[1]
            margins.append(float(values[selected_index].route_score - values[alternate].route_score))
    thresholds = {0.0}
    for margin in margins:
        thresholds.add(float(margin + 1e-6))
    if margins:
        thresholds.add(float(max(margins) + 1e-6))
    reports = []
    for threshold in sorted(thresholds):
        report = coverage_report(bundles, scores, primary_fn, threshold)
        report["threshold"] = threshold
        report["feasible"] = bool(report["gold_coverage"] >= target_coverage)
        reports.append(report)
    feasible = [report for report in reports if report["feasible"]]
    selected = min(
        feasible or reports,
        key=lambda report: (report["mean_selected_candidates"], report["threshold"]),
    )
    selected = dict(selected)
    selected["target_coverage"] = target_coverage
    selected["candidate_thresholds"] = len(reports)
    selected["feasible_count"] = len(feasible)
    return selected


def make_rows(
    bundles: list[Any],
    scores: dict[str, list[Any]],
    answers: dict[tuple[str, str], str],
    primary_fn: Callable[[Any, dict[str, list[Any]]], tuple[int, ...]] | None,
    margin_threshold: float | None,
    *,
    method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle_index, bundle in enumerate(bundles, start=1):
        if primary_fn is None:
            choices = tuple(tuple() for _ in bundle.groups)
        else:
            primary = primary_fn(bundle, scores)
            choices = expanded_choices(bundle, scores, primary, float(margin_threshold or 0.0))
        for group_index, (group, selected_indices) in enumerate(
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
            all_candidates = [v1.evidence_json(candidate) for candidate in group.candidates]
            canonical = canonical_indices(group)
            if primary_fn is None:
                ordered_indices = canonical
            else:
                ordered_indices = list(selected_indices) + [
                    index for index in canonical if index not in selected_indices
                ]
            selected_candidates = [all_candidates[index] for index in ordered_indices]
            rows.append(
                {
                    "request_id": f"c3c4-fair-order-{bundle_index:04d}-{group_index:02d}",
                    "split": "c3_c4_joint_fair_order_v1",
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
                        "risk_types": v1.risk_values(group),
                    },
                    "routing": {
                        "method": method,
                        "evidence_policy": "preserve_all_fair_order",
                        "priority_indices": list(selected_indices),
                        "selected_indices": ordered_indices,
                        "selected_evidence_ids": [item["evidence_id"] for item in selected_candidates],
                        "source_group_id": group.group_id,
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, default=Path("data/processed/router_v1/train/source_swap.jsonl"))
    parser.add_argument("--dev-input", type=Path, default=Path("data/processed/c3_blind_v1/source_swap.jsonl"))
    parser.add_argument("--answers-input", type=Path, default=Path("data/raw/averitec/dev.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit-bundles", type=int, default=100)
    parser.add_argument("--dev-limit-bundles", type=int, default=20)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--target-gold-coverage", type=float, default=0.90)
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
    if not 0.0 < args.target_gold_coverage <= 1.0:
        raise ValueError("target gold coverage must be in (0, 1]")

    c3._patch_feature_functions()
    train_bundles = c3.base.load_bundles(args.train_input, args.train_limit_bundles)
    dev_bundles = c3.base.load_bundles(args.dev_input, args.dev_limit_bundles)
    calibration_size = max(1, int(round(len(train_bundles) * args.calibration_fraction)))
    if calibration_size >= len(train_bundles):
        raise ValueError("training limit is too small for calibration")
    fit_bundles = train_bundles[:-calibration_size]
    calibration_bundles = train_bundles[-calibration_size:]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer", local_files_only=True)
    model = MultiTaskProvenanceRouter.from_pretrained(args.checkpoint / "backbone", dropout=0.1)
    model.heads.load_state_dict(torch.load(args.checkpoint / "router_heads.pt", map_location="cpu", weights_only=True))
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()

    score_kwargs = {"batch_size": args.group_batch_size, "max_length": args.max_length, "device": device}
    fit_scores = c3.base.route_scores(model, tokenizer, fit_bundles, **score_kwargs)
    calibration_scores = c3.base.route_scores(model, tokenizer, calibration_bundles, **score_kwargs)
    dev_scores = c3.base.route_scores(model, tokenizer, dev_bundles, **score_kwargs)
    weight = c3_structured.implementation.fit_assignment_head(
        fit_bundles, fit_scores, args.candidate_limit,
        epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed,
    )
    c2_primary = primary_independent
    c3_primary = lambda bundle, scores: primary_structured(bundle, scores, weight, args.candidate_limit)
    c2_policy = calibrate_margin(calibration_bundles, calibration_scores, c2_primary, args.target_gold_coverage)
    c3_policy = calibrate_margin(calibration_bundles, calibration_scores, c3_primary, args.target_gold_coverage)
    answers = v1.load_answers(args.answers_input)

    raw_rows = make_rows(dev_bundles, dev_scores, answers, None, None, method="raw_all_candidates")
    c2_rows = make_rows(dev_bundles, dev_scores, answers, c2_primary, float(c2_policy["threshold"]), method="c2_temporal_independent_preserve_all")
    c3_rows = make_rows(dev_bundles, dev_scores, answers, c3_primary, float(c3_policy["threshold"]), method="c3_structured_assignment_preserve_all")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"raw": args.output_dir / "raw_queue.jsonl", "c2": args.output_dir / "c2_queue.jsonl", "c3": args.output_dir / "c3_queue.jsonl"}
    for key, rows in [("raw", raw_rows), ("c2", c2_rows), ("c3", c3_rows)]:
        v1.write_jsonl(paths[key], rows)
    manifest = {
        "status": "complete",
        "version": "c3_c4_joint_fair_order_v1",
        "train_bundles": len(train_bundles),
        "fit_bundles": len(fit_bundles),
        "calibration_bundles": len(calibration_bundles),
        "dev_bundles": len(dev_bundles),
        "dev_groups": len(raw_rows),
        "target_gold_coverage": args.target_gold_coverage,
        "selection_policies": {"c2": c2_policy, "c3": c3_policy},
        "methods": {key: str(path) for key, path in paths.items()},
        "learned_weights": weight.tolist(),
        "sha256": {key: sha256_file(path) for key, path in paths.items()},
        "policy": [
            "Development smoke only; no calibration after API outputs.",
            "Candidate order is a stable hash order independent of gold_route; C2/C3 priority is applied after canonical ordering.",
            "Raw, C2, and C3 use the same claim groups and frozen generator contract.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
