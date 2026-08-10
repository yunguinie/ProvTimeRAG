"""Evaluate a frozen off-the-shelf reranker on publisher Source-Swap.

This runner deliberately performs no fitting or model selection.  It supports
both a content-only diagnostic view and the publisher-visible input contract
used by the final C2 router.  Sentence-Transformers CrossEncoder provides a
common inference interface for MiniLM, BGE, Mixedbread, and Qwen3 rerankers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

from provtimerag.data import ClaimBundle, ClaimRouteGroup, read_bundles_jsonl
from provtimerag.evaluation import evaluate_rankings
from provtimerag.routing.multitask_data import candidate_text, request_text


PUBLISHER_INSTRUCTION = (
    "Rank evidence for publisher attribution. Prefer the candidate whose "
    "observable publisher identity, URL/domain, source role, and document "
    "metadata are compatible with the question and atomic claim. Do not infer "
    "correctness from candidate position."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observable_pair(
    group: ClaimRouteGroup,
    candidate_index: int,
    *,
    input_view: str,
) -> tuple[str, str]:
    """Return label-free model inputs under an explicit visibility contract."""

    query = f"{PUBLISHER_INSTRUCTION}\n\n{request_text(group)}"
    evidence = group.candidates[candidate_index].evidence
    if input_view == "content_only":
        document = f"Evidence: {evidence.text}"
    elif input_view == "publisher_visible":
        document = candidate_text(evidence, include_source_url=True)
    else:
        raise ValueError(f"unknown input view: {input_view}")
    return query, document


def tie_key(group_id: str, evidence_id: str) -> str:
    return hashlib.sha256(f"{group_id}\0{evidence_id}".encode("utf-8")).hexdigest()


def ranking_from_scores(
    group: ClaimRouteGroup,
    scores: Sequence[float],
) -> tuple[str, ...]:
    if len(scores) != len(group.candidates):
        raise ValueError("one score is required for every candidate")
    indexed = list(enumerate(scores))
    indexed.sort(
        key=lambda item: (
            -float(item[1]),
            tie_key(
                group.group_id,
                group.candidates[item[0]].evidence.evidence_id,
            ),
        )
    )
    return tuple(
        group.candidates[index].evidence.evidence_id for index, _ in indexed
    )


def score_groups(
    model: Any,
    tokenizer: Any,
    groups: Sequence[ClaimRouteGroup],
    *,
    input_view: str,
    batch_size: int,
    max_length: int,
    device: Any,
) -> tuple[dict[str, tuple[str, ...]], list[dict[str, Any]]]:
    import torch

    pairs = [
        observable_pair(group, index, input_view=input_view)
        for group in groups
        for index in range(len(group.candidates))
    ]
    flat: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        inputs = tokenizer(
            [query for query, _ in batch],
            [document for _, document in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, return_dict=True).logits.float()
        if logits.ndim == 1:
            scores = logits
        elif logits.shape[-1] == 1:
            scores = logits[:, 0]
        elif logits.shape[-1] == 2:
            scores = logits[:, 1]
        else:
            raise RuntimeError(
                f"expected one or two classifier logits, received {tuple(logits.shape)}"
            )
        flat.extend(float(value) for value in scores.cpu().tolist())
        processed = min(start + len(batch), len(pairs))
        print(f"PROGRESS candidates={processed}/{len(pairs)}", flush=True)

    rankings: dict[str, tuple[str, ...]] = {}
    predictions: list[dict[str, Any]] = []
    cursor = 0
    for group in groups:
        count = len(group.candidates)
        scores = [float(value) for value in flat[cursor : cursor + count]]
        cursor += count
        ranking = ranking_from_scores(group, scores)
        rankings[group.group_id] = ranking
        predictions.append(
            {
                "group_id": group.group_id,
                "candidate_ids": [
                    candidate.evidence.evidence_id for candidate in group.candidates
                ],
                "scores": scores,
                "ranking": list(ranking),
            }
        )
    if cursor != len(flat):
        raise RuntimeError("model score count does not match candidate count")
    return rankings, predictions

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input-view",
        choices=("content_only", "publisher_visible"),
        default="publisher_visible",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    bundles: list[ClaimBundle] = list(read_bundles_jsonl(args.input))
    groups = [group for bundle in bundles for group in bundle.groups]
    if not groups:
        raise RuntimeError("input contains no route groups")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = time.perf_counter()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device).eval()
    rankings, predictions = score_groups(
        model,
        tokenizer,
        groups,
        input_view=args.input_view,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    elapsed = time.perf_counter() - started
    abstentions = {group.group_id: False for group in groups}
    metrics = evaluate_rankings(bundles, rankings, abstentions)

    report = {
        "status": "complete",
        "method": "frozen_off_the_shelf_cross_encoder_v1",
        "evaluation_contract": "no_fitting_no_model_selection_label_independent_ties",
        "model": args.model,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "input_view": args.input_view,
        "source_url_included": args.input_view == "publisher_visible",
        "instruction": PUBLISHER_INSTRUCTION,
        "bundles": len(bundles),
        "groups": len(groups),
        "candidates": sum(len(group.candidates) for group in groups),
        "device": args.device,
        "dtype": str(dtype),
        "max_length": args.max_length,
        "elapsed_seconds": elapsed,
        "metrics": metrics.__dict__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prediction_path = args.output.with_suffix(".predictions.jsonl")
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({**report, "predictions": str(prediction_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()





