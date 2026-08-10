"""Train the balanced multi-task Provenance-State Router on frozen supervision."""

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
from transformers import AutoTokenizer  # type: ignore[import-untyped]

from provtimerag.data import read_bundles_jsonl
from provtimerag.routing.multitask import multitask_router_loss
from provtimerag.routing.multitask_data import (
    TASK_INSUFFICIENT,
    TASK_SOURCE,
    TASK_TEMPORAL,
    balanced_epoch,
    bundle_examples,
    flatten_batch,
)
from provtimerag.routing.multitask_model import MultiTaskProvenanceRouter


TRAIN_PATHS = {
    TASK_SOURCE: Path("data/processed/router_v1/train/source_swap.jsonl"),
    TASK_TEMPORAL: Path("data/interim/hoh/train.jsonl"),
    TASK_INSUFFICIENT: Path("data/processed/router_v1/train/insufficient.jsonl"),
}
DEV_PATHS = {
    TASK_SOURCE: Path("data/processed/router_v1/dev/source_swap.jsonl"),
    TASK_TEMPORAL: Path("data/interim/hoh/dev.jsonl"),
    TASK_INSUFFICIENT: Path("data/processed/router_v1/dev/insufficient.jsonl"),
}


def load_task(path: Path, task: str, limit: int | None) -> list[Any]:
    bundles = read_bundles_jsonl(path)
    selected = bundles if limit is None else islice(bundles, limit)
    return bundle_examples(selected, task)


def encode(
    tokenizer: Any,
    requests: list[str],
    candidates: list[str],
    *,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    values = tokenizer(
        requests,
        candidates,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: tensor.to(device) for key, tensor in values.items()}


def binary_metrics(targets: list[bool], predictions: list[bool]) -> dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("binary targets and predictions must align")
    true_positive = sum(target and prediction for target, prediction in zip(targets, predictions))
    false_positive = sum(not target and prediction for target, prediction in zip(targets, predictions))
    false_negative = sum(target and not prediction for target, prediction in zip(targets, predictions))
    true_negative = sum(not target and not prediction for target, prediction in zip(targets, predictions))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "support": len(targets),
        "positive_support": sum(targets),
        "predicted_positive": sum(predictions),
        "accuracy": (true_positive + true_negative) / len(targets) if targets else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "tn": true_negative,
        },
    }


def evaluate(
    model: MultiTaskProvenanceRouter,
    tokenizer: Any,
    examples_by_task: dict[str, list[Any]],
    *,
    group_batch_size: int,
    max_length: int,
    device: torch.device,
    abstention_threshold: float,
    seen_sources: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    model.eval()
    with torch.inference_mode():
        for task, examples in examples_by_task.items():
            route_hits = answerable = abstention_hits = 0
            abstention_targets: list[bool] = []
            abstention_predictions: list[bool] = []
            coverage_hits = {"seen": 0, "unseen": 0}
            coverage_totals = {"seen": 0, "unseen": 0}
            losses: list[float] = []
            for start in range(0, len(examples), group_batch_size):
                batch = examples[start : start + group_batch_size]
                requests, candidates, targets = flatten_batch(batch, device)
                encoded = encode(
                    tokenizer, requests, candidates,
                    max_length=max_length, device=device,
                )
                outputs = model(encoded, targets.group_index, len(batch))
                losses.append(float(multitask_router_loss(outputs, targets).total.cpu()))
                for group, example in enumerate(batch):
                    selected = targets.group_index == group
                    if any(example.gold_route):
                        local_scores = outputs.route_logits[selected]
                        predicted = int(torch.argmax(local_scores).cpu())
                        route_hits += int(example.gold_route[predicted])
                        answerable += 1
                        if task == TASK_SOURCE:
                            gold_sources = {
                                source_id
                                for source_id, gold in zip(
                                    example.candidate_source_ids,
                                    example.gold_route,
                                    strict=True,
                                )
                                if gold
                            }
                            coverage = "seen" if gold_sources & seen_sources else "unseen"
                            coverage_totals[coverage] += 1
                            coverage_hits[coverage] += int(example.gold_route[predicted])
                    abstain = bool(
                        torch.sigmoid(outputs.abstention_logits[group]).cpu()
                        >= abstention_threshold
                    )
                    abstention_hits += int(abstain == example.should_abstain)
                    abstention_targets.append(example.should_abstain)
                    abstention_predictions.append(abstain)
            result[task] = {
                "groups": len(examples),
                "mean_loss": sum(losses) / max(1, len(losses)),
                "top1_accuracy": route_hits / answerable if answerable else None,
                "answerable_groups": answerable,
                "abstention_accuracy": abstention_hits / max(1, len(examples)),
                "abstention": binary_metrics(
                    abstention_targets, abstention_predictions
                ),
            }
            if task == TASK_SOURCE:
                result[task]["source_coverage"] = {
                    name: {
                        "groups": coverage_totals[name],
                        "top1_accuracy": (
                            coverage_hits[name] / coverage_totals[name]
                            if coverage_totals[name]
                            else None
                        ),
                    }
                    for name in ("seen", "unseen")
                }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--output", type=Path, default=Path("results/router_v3/multitask_smoke"))
    parser.add_argument("--train-limit-per-task", type=int, default=200)
    parser.add_argument("--dev-limit-per-task", type=int, default=100)
    parser.add_argument("--groups-per-task", type=int, default=200)
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

    if not 0 < args.abstention_threshold < 1:
        raise ValueError("abstention threshold must be between zero and one")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = {
        task: load_task(path, task, args.train_limit_per_task)
        for task, path in TRAIN_PATHS.items()
    }
    dev = {
        task: load_task(path, task, args.dev_limit_per_task)
        for task, path in DEV_PATHS.items()
    }
    epoch_examples = balanced_epoch(
        train, groups_per_task=args.groups_per_task, seed=args.seed
    )
    seen_sources = {
        source_id
        for example in train[TASK_SOURCE]
        for source_id, gold in zip(
            example.candidate_source_ids, example.gold_route, strict=True
        )
        if gold
    }
    if args.dry_run:
        sample = epoch_examples[: min(3, len(epoch_examples))]
        requests, candidates, targets = flatten_batch(sample)
        report = {
            "status": "dry_run_ok", "api_called": False, "model_loaded": False,
            "train_groups": {task: len(rows) for task, rows in train.items()},
            "dev_groups": {task: len(rows) for task, rows in dev.items()},
            "balanced_epoch_groups": len(epoch_examples),
            "sample_candidates": len(candidates),
            "sample_task_masks": {
                "source": int(targets.source_mask.sum()),
                "temporal": int(targets.temporal_mask.sum()),
                "version": int(targets.version_mask.sum()),
            },
            "seen_source_ids": len(seen_sources),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MultiTaskProvenanceRouter.from_pretrained(
        args.model, dropout=args.dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    use_amp = args.bf16 and device.type == "cuda"
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        epoch_examples = balanced_epoch(
            train, groups_per_task=args.groups_per_task, seed=args.seed + epoch
        )
        batches = math.ceil(len(epoch_examples) / args.group_batch_size)
        running = {name: 0.0 for name in ("total", "route", "source", "temporal", "version", "abstention")}
        for batch_index, start in enumerate(range(0, len(epoch_examples), args.group_batch_size), 1):
            batch = epoch_examples[start : start + args.group_batch_size]
            requests, candidates, targets = flatten_batch(batch, device)
            encoded = encode(tokenizer, requests, candidates, max_length=args.max_length, device=device)
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
                    print(
                        f"PROGRESS epoch={epoch + 1}/{args.epochs} step={global_step} "
                        f"loss={float(losses.total.detach().cpu()):.4f}", flush=True,
                    )
        history.append({
            "epoch": epoch + 1,
            "optimizer_steps": global_step,
            **{name: value / max(1, batches) for name, value in running.items()},
        })

    metrics = evaluate(
        model, tokenizer, dev,
        group_batch_size=args.group_batch_size,
        max_length=args.max_length, device=device,
        abstention_threshold=args.abstention_threshold,
        seen_sources=seen_sources,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_router(args.output / "checkpoint")
    tokenizer.save_pretrained(args.output / "checkpoint" / "tokenizer")
    report = {
        "status": "complete", "model": args.model, "device": str(device),
        "train_groups": {task: len(rows) for task, rows in train.items()},
        "dev_groups": {task: len(rows) for task, rows in dev.items()},
        "groups_per_task": args.groups_per_task, "epochs": args.epochs,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history, "dev": metrics,
        "checkpoint": str(args.output / "checkpoint"),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
