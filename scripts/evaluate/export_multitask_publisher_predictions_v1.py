"""Export frozen C2 per-group predictions for paired statistical analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from provtimerag.data import read_bundles_jsonl


def frequency_bucket(count: int) -> str:
    if count == 0:
        return "unseen"
    if count <= 5:
        return "tail_1_5"
    if count <= 20:
        return "mid_6_20"
    return "head_21_plus"


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    from provtimerag.routing.multitask_data import TASK_SOURCE, flatten_batch
    from provtimerag.routing.multitask_model import MultiTaskProvenanceRouter
    from scripts.train import train_multitask_provenance_router as legacy
    from scripts.train.train_multitask_publisher_router_v3 import (
        load_task,
        sha256_file,
        stable_best_index,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--abstention-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    train_examples = load_task(args.source_train, TASK_SOURCE, None)
    examples = load_task(args.input, TASK_SOURCE, None)
    train_frequency = Counter(
        source
        for example in train_examples
        for source, gold in zip(
            example.candidate_source_ids, example.gold_route, strict=True
        )
        if gold
    )
    bundle_by_group: dict[str, str] = {}
    bundle_size: dict[str, int] = {}
    evidence_ids: dict[str, list[str]] = {}
    for bundle in read_bundles_jsonl(args.input):
        bundle_size[bundle.bundle_id] = len(bundle.groups)
        for group in bundle.groups:
            bundle_by_group[group.group_id] = bundle.bundle_id
            evidence_ids[group.group_id] = [
                candidate.evidence.evidence_id for candidate in group.candidates
            ]

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint / "tokenizer", local_files_only=True
    )
    model = MultiTaskProvenanceRouter.from_pretrained(
        args.checkpoint / "backbone", dropout=0.1
    )
    state = torch.load(
        args.checkpoint / "router_heads.pt", map_location="cpu", weights_only=True
    )
    model.heads.load_state_dict(state)
    model.to(device=device, dtype=dtype).eval()

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(examples), args.group_batch_size):
            batch = examples[start : start + args.group_batch_size]
            requests, candidates, targets = flatten_batch(batch, device)
            encoded = legacy.encode(
                tokenizer,
                requests,
                candidates,
                max_length=args.max_length,
                device=device,
            )
            outputs = model(encoded, targets.group_index, len(batch))
            for group_index, example in enumerate(batch):
                selected_mask = targets.group_index == group_index
                local_scores = outputs.route_logits[selected_mask]
                predicted, tied = stable_best_index(example, local_scores)
                gold_indices = [
                    index for index, gold in enumerate(example.gold_route) if gold
                ]
                gold_sources = sorted(
                    source
                    for source, gold in zip(
                        example.candidate_source_ids,
                        example.gold_route,
                        strict=True,
                    )
                    if gold
                )
                maximum_frequency = max(
                    (train_frequency[source] for source in gold_sources), default=0
                )
                abstain_probability = float(
                    torch.sigmoid(outputs.abstention_logits[group_index]).float().cpu()
                )
                ids = evidence_ids[example.group_id]
                bundle_id = bundle_by_group[example.group_id]
                rows.append(
                    {
                        "group_id": example.group_id,
                        "bundle_id": bundle_id,
                        "bundle_size": bundle_size[bundle_id],
                        "candidate_ids": ids,
                        "candidate_source_ids": list(example.candidate_source_ids),
                        "gold_indices": gold_indices,
                        "gold_candidate_ids": [ids[index] for index in gold_indices],
                        "gold_sources": gold_sources,
                        "publisher_frequency": maximum_frequency,
                        "publisher_frequency_bucket": frequency_bucket(maximum_frequency),
                        "scores": local_scores.detach().float().cpu().tolist(),
                        "predicted_index": predicted,
                        "predicted_candidate_id": ids[predicted],
                        "hit": bool(example.gold_route[predicted]),
                        "exact_score_tie": tied,
                        "should_abstain": example.should_abstain,
                        "abstention_probability": abstain_probability,
                        "abstain": abstain_probability >= args.abstention_threshold,
                    }
                )
            processed = min(start + len(batch), len(examples))
            print(f"PROGRESS groups={processed}/{len(examples)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "status": "complete",
        "method": "frozen_c2_prediction_export_v1",
        "checkpoint": str(args.checkpoint),
        "source_train": str(args.source_train),
        "input": str(args.input),
        "output": str(args.output),
        "groups": len(rows),
        "top1_accuracy": sum(row["hit"] for row in rows) / len(rows),
        "exact_score_ties": sum(row["exact_score_tie"] for row in rows),
        "dtype": str(dtype),
        "sha256": {
            "source_train": sha256_file(args.source_train),
            "input": sha256_file(args.input),
        },
    }
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
