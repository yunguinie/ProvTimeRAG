"""Evaluate the frozen Qwen3 Reranker on publisher Source-Swap."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from provtimerag.data import ClaimBundle, ClaimRouteGroup, read_bundles_jsonl
from provtimerag.evaluation import evaluate_rankings
from provtimerag.routing.multitask_data import candidate_text, request_text
from scripts.evaluate.run_c3_strong_cross_encoder_baseline_v1 import (
    PUBLISHER_INSTRUCTION,
    ranking_from_scores,
    sha256_file,
)


PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements '
    'based on the Query and the Instruct provided. Note that the answer can '
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def model_text(group: ClaimRouteGroup, candidate_index: int, input_view: str) -> str:
    evidence = group.candidates[candidate_index].evidence
    if input_view == "content_only":
        document = f"Evidence: {evidence.text}"
    elif input_view == "publisher_visible":
        document = candidate_text(evidence, include_source_url=True)
    else:
        raise ValueError(f"unknown input view: {input_view}")
    return (
        f"<Instruct>: {PUBLISHER_INSTRUCTION}\n"
        f"<Query>: {request_text(group)}\n"
        f"<Document>: {document}"
    )


def score_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    device: Any,
) -> list[float]:
    import torch

    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    true_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_id = tokenizer("no", add_special_tokens=False).input_ids[0]
    budget = max_length - len(prefix) - len(suffix)
    if budget <= 0:
        raise ValueError("max length is shorter than the Qwen3 prompt wrapper")

    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=False,
            truncation=True,
            max_length=budget,
            return_attention_mask=False,
        )["input_ids"]
        wrapped = [prefix + ids + suffix for ids in batch]
        inputs = tokenizer.pad(
            {"input_ids": wrapped}, padding=True, return_tensors="pt"
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, return_dict=True).logits[:, -1, :].float()
        binary = torch.stack([logits[:, false_id], logits[:, true_id]], dim=1)
        scores.extend(binary.log_softmax(dim=1)[:, 1].exp().cpu().tolist())
        processed = min(start + len(batch), len(texts))
        print(f"PROGRESS candidates={processed}/{len(texts)}", flush=True)
    return [float(score) for score in scores]


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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bundles: list[ClaimBundle] = list(read_bundles_jsonl(args.input))
    groups = [group for bundle in bundles for group in bundle.groups]
    texts = [
        model_text(group, index, args.input_view)
        for group in groups
        for index in range(len(group.candidates))
    ]
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    flat_scores = score_texts(
        model,
        tokenizer,
        texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    rankings: dict[str, tuple[str, ...]] = {}
    predictions: list[dict[str, Any]] = []
    cursor = 0
    for group in groups:
        count = len(group.candidates)
        scores = flat_scores[cursor : cursor + count]
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
    abstentions = {group.group_id: False for group in groups}
    metrics = evaluate_rankings(bundles, rankings, abstentions)
    report = {
        "status": "complete",
        "method": "frozen_qwen3_reranker_off_the_shelf_v1",
        "evaluation_contract": "official_yes_no_logits_no_fitting_label_independent_ties",
        "model": args.model,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "input_view": args.input_view,
        "source_url_included": args.input_view == "publisher_visible",
        "instruction": PUBLISHER_INSTRUCTION,
        "bundles": len(bundles),
        "groups": len(groups),
        "candidates": len(texts),
        "device": args.device,
        "dtype": str(dtype),
        "max_length": args.max_length,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics.__dict__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prediction_path = args.output.with_suffix(".predictions.jsonl")
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({**report, "predictions": str(prediction_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
