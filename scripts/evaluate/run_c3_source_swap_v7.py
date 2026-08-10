"""C3 Source Swap evaluator with source-only learned compatibility.

Source Swap is a source-attribution stress test. Its version identifiers are
synthetic/unavailable, so this evaluator does not train a compatibility model
on those identifiers and only scores version consistency on bundles for which
the evidence carries an informative time/version signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from provtimerag.data import EvidenceRecord
from provtimerag.routing.multitask_model import MultiTaskProvenanceRouter
from scripts.evaluate import run_c3_global_consistency as CORE
from scripts.evaluate import run_c3_source_swap_v2 as base


# Version fields are deliberately excluded from the learned Source Swap
# compatibility objective. They are unavailable/synthetic in this slice.
FEATURE_NAMES = (
    "bias",
    "same_source",
    "same_document",
    "same_entity",
    "same_event",
    "same_role",
)


def informative_version(evidence: EvidenceRecord) -> bool:
    metadata = evidence.metadata or {}
    label = metadata.get("version_label")
    return (
        label not in (None, "", "unavailable")
        or evidence.valid_from is not None
        or evidence.valid_to is not None
        or evidence.publication_time is not None
    )


def pair_features(a: EvidenceRecord, b: EvidenceRecord) -> np.ndarray:
    return np.asarray(
        [
            1.0,
            float(a.source_id == b.source_id),
            float(a.document_id == b.document_id),
            float(a.entity_id is not None and a.entity_id == b.entity_id),
            float(a.event_id is not None and a.event_id == b.event_id),
            float(a.source_role == b.source_role),
        ],
        dtype=np.float32,
    )


def _patch_feature_functions() -> None:
    """Patch the shared decoder/pair-row code with the six source features."""

    CORE.FEATURE_NAMES = FEATURE_NAMES
    CORE.pair_features = pair_features
    base.FEATURE_NAMES = FEATURE_NAMES
    base.pair_features = pair_features


def _version_info(bundle: Any, selected: list[Any]) -> tuple[bool, bool, bool]:
    """Return (applicable, gold_mix, predicted_mix) for a bundle."""

    all_items = [candidate for group in bundle.groups for candidate in group.candidates]
    applicable = all(informative_version(item.evidence) for item in all_items)
    if not applicable:
        return False, False, False
    gold_sources = {
        candidate.evidence.source_id
        for group in bundle.groups
        for candidate in group.candidates
        if candidate.gold_route
    }
    pred_sources = {item.evidence.source_id for item in selected}
    gold_versions = {
        (candidate.evidence.source_id, candidate.evidence.version_id)
        for group in bundle.groups
        for candidate in group.candidates
        if candidate.gold_route
    }
    pred_versions = {(item.evidence.source_id, item.evidence.version_id) for item in selected}
    return (
        True,
        len(gold_versions) > 1 and len(gold_sources) == 1,
        len(pred_versions) > 1 and len(pred_sources) == 1,
    )


def evaluate_decoder(
    bundles: list[Any],
    scores: dict[str, list[Any]],
    weights: np.ndarray,
    *,
    beam_size: int,
    candidate_limit: int,
) -> dict[str, Any]:
    exact = source_conflation = source_undercoverage = selected_groups = 0
    gold_multi = predicted_multi = 0
    temporal_inconsistency = 0
    version_applicable = gold_version_mix = predicted_version_mix = 0
    for bundle in bundles:
        chosen = base.beam_decode(
            bundle, scores, weights, beam_size=beam_size, candidate_limit=candidate_limit
        )
        selected = [
            scores[group.group_id][index]
            for group, index in zip(bundle.groups, chosen, strict=True)
        ]
        exact += int(all(item.gold for item in selected))
        selected_groups += len(selected)
        gold_sources = {
            candidate.evidence.source_id
            for group in bundle.groups
            for candidate in group.candidates
            if candidate.gold_route
        }
        pred_sources = {item.evidence.source_id for item in selected}
        gold_is_multi = len(gold_sources) > 1
        pred_is_multi = len(pred_sources) > 1
        gold_multi += int(gold_is_multi)
        predicted_multi += int(pred_is_multi)
        source_conflation += int(pred_is_multi and not gold_is_multi)
        source_undercoverage += int(gold_is_multi and not pred_is_multi)
        temporal_inconsistency += int(
            any(
                base.temporal_conflict(left.evidence, right.evidence)
                for left in selected
                for right in selected
            )
        )
        applicable, gold_mix, predicted_mix = _version_info(bundle, selected)
        if applicable:
            version_applicable += 1
            gold_version_mix += int(gold_mix)
            predicted_version_mix += int(predicted_mix)
    count = len(bundles)
    return {
        "bundles": count,
        "bundle_exact_match": exact / count if count else None,
        "gold_multi_source_rate": gold_multi / count if count else None,
        "predicted_multi_source_rate": predicted_multi / count if count else None,
        "source_conflation_rate": source_conflation / count if count else None,
        "source_undercoverage_rate": source_undercoverage / count if count else None,
        "temporal_inconsistency_rate": temporal_inconsistency / count if count else None,
        "version_metrics_applicable_bundles": version_applicable,
        "gold_version_mixing_rate": gold_version_mix / version_applicable
        if version_applicable
        else None,
        "predicted_version_mixing_rate": predicted_version_mix / version_applicable
        if version_applicable
        else None,
        "selected_groups": selected_groups,
    }


def main() -> None:
    _patch_feature_functions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, default=Path("data/processed/router_v1/train/source_swap.jsonl"))
    parser.add_argument("--dev-input", type=Path, default=Path("data/processed/router_v1/dev/source_swap.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit-bundles", type=int, default=20)
    parser.add_argument("--dev-limit-bundles", type=int, default=20)
    parser.add_argument("--group-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    train_bundles = base.load_bundles(args.train_input, args.train_limit_bundles)
    dev_bundles = base.load_bundles(args.dev_input, args.dev_limit_bundles)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer", local_files_only=True)
    model = MultiTaskProvenanceRouter.from_pretrained(args.checkpoint / "backbone", dropout=0.1)
    model.heads.load_state_dict(
        torch.load(args.checkpoint / "router_heads.pt", map_location="cpu", weights_only=True)
    )
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    train_scores = base.route_scores(
        model, tokenizer, train_bundles, batch_size=args.group_batch_size,
        max_length=args.max_length, device=device,
    )
    dev_scores = base.route_scores(
        model, tokenizer, dev_bundles, batch_size=args.group_batch_size,
        max_length=args.max_length, device=device,
    )
    features, labels = base.pair_rows(train_bundles)
    learned = CORE.fit_logistic(features, labels)
    independent = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    ordinary_beam = evaluate_decoder(
        dev_bundles, dev_scores, independent, beam_size=args.beam_size,
        candidate_limit=args.candidate_limit,
    )
    report = {
        "status": "complete",
        "compatibility_scope": "source_attribution_only",
        "version_signal_policy": "score_only_when_informative",
        "train_bundles": len(train_bundles),
        "dev_bundles": len(dev_bundles),
        "train_pair_examples": len(labels),
        "train_positive_pairs": int(labels.sum()),
        "train_negative_pairs": int((labels == 0).sum()),
        "feature_names": FEATURE_NAMES,
        "learned_weights": learned.tolist(),
        "metrics": {
            "independent_top1": evaluate_decoder(
                dev_bundles, dev_scores, independent, beam_size=1,
                candidate_limit=args.candidate_limit,
            ),
            "ordinary_beam": ordinary_beam,
            "learned_compatibility": evaluate_decoder(
                dev_bundles, dev_scores, learned, beam_size=args.beam_size,
                candidate_limit=args.candidate_limit,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
