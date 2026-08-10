"""Train compute-matched publisher Router ablations on the frozen v3 contract."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer

from provtimerag.data import read_bundles_jsonl
from provtimerag.routing.multitask import multitask_router_loss
from provtimerag.routing.multitask_data import (
    TASK_INSUFFICIENT, TASK_SOURCE, TASK_TEMPORAL, TASKS,
    RouterGroupExample, balanced_epoch, bundle_examples,
    deterministic_order, flatten_batch,
)
from provtimerag.routing.multitask_model import MultiTaskProvenanceRouter
from scripts.train import train_multitask_provenance_router as legacy
from scripts.train.train_multitask_publisher_router_v3 import evaluate, sha256_file

MODES = ("multitask", "source_only_compute_matched")


def load_task_view(path: Path, task: str, limit: int | None, *, include_source_url: bool) -> list[RouterGroupExample]:
    bundles = read_bundles_jsonl(path)
    selected = bundles if limit is None else islice(bundles, limit)
    return bundle_examples(selected, task, include_source_url=include_source_url and task == TASK_SOURCE)


def epoch_examples(examples_by_task: dict[str, list[RouterGroupExample]], *, mode: str, groups_per_task: int, seed: int) -> list[RouterGroupExample]:
    if mode == "multitask":
        return balanced_epoch(examples_by_task, groups_per_task=groups_per_task, seed=seed)
    if mode != "source_only_compute_matched":
        raise ValueError(f"unknown ablation mode: {mode}")
    if groups_per_task <= 0 or not examples_by_task.get(TASK_SOURCE):
        raise ValueError("source-only ablation requires positive groups and source examples")
    ordered = deterministic_order(examples_by_task[TASK_SOURCE], seed)
    selected = [ordered[index % len(ordered)] for index in range(groups_per_task * len(TASKS))]
    random.Random(seed).shuffle(selected)
    return selected


def source_url_contract(examples: list[RouterGroupExample], *, expected: bool) -> bool:
    observed = all("Source URL:" in text and "Source domain:" in text for example in examples for text in example.candidate_texts)
    return observed == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-dev", type=Path, required=True)
    parser.add_argument("--temporal-train", type=Path, required=True)
    parser.add_argument("--temporal-dev", type=Path, required=True)
    parser.add_argument("--insufficient-train", type=Path, required=True)
    parser.add_argument("--insufficient-dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--exclude-source-url", action="store_true")
    parser.add_argument("--train-limit-per-task", type=int)
    parser.add_argument("--dev-limit-per-task", type=int)
    parser.add_argument("--groups-per-task", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--abstention-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    include_source_url = not args.exclude_source_url
    train_paths = {TASK_SOURCE: args.source_train, TASK_TEMPORAL: args.temporal_train, TASK_INSUFFICIENT: args.insufficient_train}
    dev_paths = {TASK_SOURCE: args.source_dev, TASK_TEMPORAL: args.temporal_dev, TASK_INSUFFICIENT: args.insufficient_dev}
    paths = {"source_train": args.source_train, "source_dev": args.source_dev, "temporal_train": args.temporal_train, "temporal_dev": args.temporal_dev, "insufficient_train": args.insufficient_train, "insufficient_dev": args.insufficient_dev}
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = {task: load_task_view(path, task, args.train_limit_per_task, include_source_url=include_source_url) for task, path in train_paths.items()}
    dev = {task: load_task_view(path, task, args.dev_limit_per_task, include_source_url=include_source_url) for task, path in dev_paths.items()}
    if not source_url_contract([*train[TASK_SOURCE], *dev[TASK_SOURCE]], expected=include_source_url):
        raise RuntimeError("source URL input view does not match the ablation contract")
    seen_sources = {source for example in train[TASK_SOURCE] for source, gold in zip(example.candidate_source_ids, example.gold_route, strict=True) if gold}
    initial_epoch = epoch_examples(train, mode=args.mode, groups_per_task=args.groups_per_task, seed=args.seed)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_ok", "method": "publisher_router_ablation_v1", "mode": args.mode, "source_url_included": include_source_url, "train_groups": {task: len(rows) for task, rows in train.items()}, "dev_groups": {task: len(rows) for task, rows in dev.items()}, "epoch_training_groups": len(initial_epoch), "source_epoch_groups": sum(example.task == TASK_SOURCE for example in initial_epoch), "input_sha256": {name: sha256_file(path) for name, path in paths.items()}}, ensure_ascii=False, indent=2))
        return

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = MultiTaskProvenanceRouter.from_pretrained(args.model, dropout=args.dropout, gradient_checkpointing=args.gradient_checkpointing).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = args.bf16 and device.type == "cuda"
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        training = epoch_examples(train, mode=args.mode, groups_per_task=args.groups_per_task, seed=args.seed + epoch)
        batches = math.ceil(len(training) / args.group_batch_size)
        running = {name: 0.0 for name in ("total", "route", "source", "temporal", "version", "abstention")}
        for batch_index, start in enumerate(range(0, len(training), args.group_batch_size), start=1):
            batch = training[start : start + args.group_batch_size]
            requests, candidates, targets = flatten_batch(batch, device)
            encoded = legacy.encode(tokenizer, requests, candidates, max_length=args.max_length, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(encoded, targets.group_index, len(batch))
                losses = multitask_router_loss(outputs, targets)
                loss = losses.total / args.gradient_accumulation
            loss.backward()
            for name in running:
                running[name] += float(getattr(losses, name).detach().cpu())
            if batch_index % args.gradient_accumulation == 0 or batch_index == batches:
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    print(f"PROGRESS epoch={epoch + 1}/{args.epochs} step={global_step} loss={float(losses.total.detach().cpu()):.4f}", flush=True)
        history.append({"epoch": epoch + 1, "optimizer_steps": global_step, "training_groups": len(training), "source_training_groups": sum(example.task == TASK_SOURCE for example in training), **{name: value / max(1, batches) for name, value in running.items()}})

    metrics = evaluate(model, tokenizer, dev, group_batch_size=args.group_batch_size, max_length=args.max_length, device=device, abstention_threshold=args.abstention_threshold, seen_sources=seen_sources)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint"
    model.save_router(checkpoint)
    tokenizer.save_pretrained(checkpoint / "tokenizer")
    report = {"status": "complete", "method": "publisher_router_ablation_v1", "data_contract": "publisher_identity_v3_leakage_controlled", "mode": args.mode, "source_url_included": include_source_url, "compute_matching": "same total epoch groups and optimizer schedule as multitask C2", "model": args.model, "device": str(device), "groups_per_task": args.groups_per_task, "epochs": args.epochs, "seed": args.seed, "elapsed_seconds": time.perf_counter() - started, "history": history, "dev": metrics, "input_sha256": {name: sha256_file(path) for name, path in paths.items()}, "checkpoint": str(checkpoint)}
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
