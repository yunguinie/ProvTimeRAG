"""Evaluate a frozen publisher Router ablation on a source-swap holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from provtimerag.routing.multitask_data import TASK_SOURCE
from provtimerag.routing.multitask_model import MultiTaskProvenanceRouter
from scripts.train.train_multitask_publisher_router_v3 import evaluate, sha256_file
from scripts.train.train_publisher_router_ablation_v1 import load_task_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-source-url", action="store_true")
    parser.add_argument("--group-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--abstention-threshold", type=float, default=0.5)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    include_source_url = not args.exclude_source_url
    train = load_task_view(args.source_train, TASK_SOURCE, None, include_source_url=include_source_url)
    holdout = load_task_view(args.input, TASK_SOURCE, None, include_source_url=include_source_url)
    seen_sources = {source for example in train for source, gold in zip(example.candidate_source_ids, example.gold_route, strict=True) if gold}
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / "tokenizer", local_files_only=True)
    model = MultiTaskProvenanceRouter.from_pretrained(args.checkpoint / "backbone", dropout=0.1)
    state = torch.load(args.checkpoint / "router_heads.pt", map_location="cpu", weights_only=True)
    model.heads.load_state_dict(state)
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)
    metrics: dict[str, Any] = evaluate(model, tokenizer, {TASK_SOURCE: holdout}, group_batch_size=args.group_batch_size, max_length=args.max_length, device=device, abstention_threshold=args.abstention_threshold, seen_sources=seen_sources)[TASK_SOURCE]
    report = {"status": "complete", "method": "publisher_router_ablation_v1_frozen_holdout", "checkpoint": str(args.checkpoint), "source_url_included": include_source_url, "source_train": str(args.source_train), "input": str(args.input), "seen_sources": len(seen_sources), "dtype": str(dtype), "input_sha256": {"source_train": sha256_file(args.source_train), "holdout": sha256_file(args.input)}, "metrics": metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
