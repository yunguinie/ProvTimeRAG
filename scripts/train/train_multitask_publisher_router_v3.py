"""Train the multitask router on leakage-controlled publisher identities."""

from __future__ import annotations

import argparse
import hashlib
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
    TASK_INSUFFICIENT,
    TASK_SOURCE,
    TASK_TEMPORAL,
    RouterGroupExample,
    balanced_epoch,
    bundle_examples,
    flatten_batch,
)
from provtimerag.routing.multitask_model import MultiTaskProvenanceRouter
from scripts.train import train_multitask_provenance_router as legacy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_task(
    path: Path, task: str, limit: int | None
) -> list[RouterGroupExample]:
    bundles = read_bundles_jsonl(path)
    selected = bundles if limit is None else islice(bundles, limit)
    return bundle_examples(
        selected,
        task,
        include_source_url=(task == TASK_SOURCE),
    )


def stable_best_index(
    example: RouterGroupExample, scores: torch.Tensor
) -> tuple[int, bool]:
    values = scores.detach().float().cpu().tolist()
    maximum = max(values)
    tied = [index for index, value in enumerate(values) if value == maximum]
    selected = min(
        tied,
        key=lambda index: hashlib.sha256(
            f"{example.group_id}\0{example.candidate_texts[index]}".encode("utf-8")
        ).hexdigest(),
    )
    return selected, len(tied) > 1


def evaluate(
    model: MultiTaskProvenanceRouter,
    tokenizer: Any,
    examples_by_task: dict[str, list[RouterGroupExample]],
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
            route_hits = answerable = abstention_hits = exact_ties = 0
            abstention_targets: list[bool] = []
            abstention_predictions: list[bool] = []
            coverage_hits = {"seen": 0, "unseen": 0}
            coverage_totals = {"seen": 0, "unseen": 0}
            losses: list[float] = []
            for start in range(0, len(examples), group_batch_size):
                batch = examples[start : start + group_batch_size]
                requests, candidates, targets = flatten_batch(batch, device)
                encoded = legacy.encode(
                    tokenizer,
                    requests,
                    candidates,
                    max_length=max_length,
                    device=device,
                )
                outputs = model(encoded, targets.group_index, len(batch))
                losses.append(float(multitask_router_loss(outputs, targets).total.cpu()))
                for group, example in enumerate(batch):
                    selected = targets.group_index == group
                    if any(example.gold_route):
                        local_scores = outputs.route_logits[selected]
                        predicted, was_tied = stable_best_index(example, local_scores)
                        hit = int(example.gold_route[predicted])
                        route_hits += hit
                        answerable += 1
                        exact_ties += int(was_tied)
                        if task == TASK_SOURCE:
                            gold_sources = {
                                source
                                for source, gold in zip(
                                    example.candidate_source_ids,
                                    example.gold_route,
                                    strict=True,
                                )
                                if gold
                            }
                            coverage = (
                                "seen" if gold_sources & seen_sources else "unseen"
                            )
                            coverage_totals[coverage] += 1
                            coverage_hits[coverage] += hit
                    abstain = bool(
                        torch.sigmoid(outputs.abstention_logits[group]).cpu()
                        >= abstention_threshold
                    )
                    abstention_hits += int(abstain == example.should_abstain)
                    abstention_targets.append(example.should_abstain)
                    abstention_predictions.append(abstain)
            task_report: dict[str, Any] = {
                "groups": len(examples),
                "mean_loss": sum(losses) / max(1, len(losses)),
                "top1_accuracy": route_hits / answerable if answerable else None,
                "answerable_groups": answerable,
                "exact_score_ties": exact_ties,
                "abstention_accuracy": abstention_hits / max(1, len(examples)),
                "abstention": legacy.binary_metrics(
                    abstention_targets, abstention_predictions
                ),
            }
            if task == TASK_SOURCE:
                task_report["source_coverage"] = {
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
            result[task] = task_report
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-dev", type=Path, required=True)
    parser.add_argument(
        "--temporal-train", type=Path, default=Path("data/interim/hoh/train.jsonl")
    )
    parser.add_argument(
        "--temporal-dev", type=Path, default=Path("data/interim/hoh/dev.jsonl")
    )
    parser.add_argument(
        "--insufficient-train",
        type=Path,
        default=Path("data/processed/router_v1/train/insufficient.jsonl"),
    )
    parser.add_argument(
        "--insufficient-dev",
        type=Path,
        default=Path("data/processed/router_v1/dev/insufficient.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
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
        raise ValueError("abstention-threshold must be in (0, 1)")

    paths = {
        "source_train": args.source_train,
        "source_dev": args.source_dev,
        "temporal_train": args.temporal_train,
        "temporal_dev": args.temporal_dev,
        "insufficient_train": args.insufficient_train,
        "insufficient_dev": args.insufficient_dev,
    }
    train_paths = {
        TASK_SOURCE: args.source_train,
        TASK_TEMPORAL: args.temporal_train,
        TASK_INSUFFICIENT: args.insufficient_train,
    }
    dev_paths = {
        TASK_SOURCE: args.source_dev,
        TASK_TEMPORAL: args.temporal_dev,
        TASK_INSUFFICIENT: args.insufficient_dev,
    }
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = {
        task: load_task(path, task, args.train_limit_per_task)
        for task, path in train_paths.items()
    }
    dev = {
        task: load_task(path, task, args.dev_limit_per_task)
        for task, path in dev_paths.items()
    }
    seen_sources = {
        source
        for example in train[TASK_SOURCE]
        for source, gold in zip(
            example.candidate_source_ids, example.gold_route, strict=True
        )
        if gold
    }
    source_url_contract = all(
        "Source URL:" in text and "Source domain:" in text
        for example in (*train[TASK_SOURCE], *dev[TASK_SOURCE])
        for text in example.candidate_texts
    )
    if not source_url_contract:
        raise RuntimeError("Source task did not expose publisher URL/domain")
    epoch_examples = balanced_epoch(
        train, groups_per_task=args.groups_per_task, seed=args.seed
    )
    if args.dry_run:
        report = {
            "status": "dry_run_ok",
            "model_loaded": False,
            "data_contract": "publisher_identity_v3_leakage_controlled",
            "source_url_included": source_url_contract,
            "train_groups": {task: len(rows) for task, rows in train.items()},
            "dev_groups": {task: len(rows) for task, rows in dev.items()},
            "balanced_epoch_groups": len(epoch_examples),
            "seen_sources": len(seen_sources),
            "sha256": {name: sha256_file(path) for name, path in paths.items()},
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = MultiTaskProvenanceRouter.from_pretrained(
        args.model,
        dropout=args.dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    use_amp = args.bf16 and device.type == "cuda"
    history: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_examples = balanced_epoch(
            train, groups_per_task=args.groups_per_task, seed=args.seed + epoch
        )
        batches = math.ceil(len(epoch_examples) / args.group_batch_size)
        running = {
            name: 0.0
            for name in (
                "total",
                "route",
                "source",
                "temporal",
                "version",
                "abstention",
            )
        }
        for batch_index, start in enumerate(
            range(0, len(epoch_examples), args.group_batch_size), start=1
        ):
            batch = epoch_examples[start : start + args.group_batch_size]
            requests, candidates, targets = flatten_batch(batch, device)
            encoded = legacy.encode(
                tokenizer,
                requests,
                candidates,
                max_length=args.max_length,
                device=device,
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                outputs = model(encoded, targets.group_index, len(batch))
                losses = multitask_router_loss(outputs, targets)
                loss = losses.total / args.gradient_accumulation
            loss.backward()
            for name in running:
                running[name] += float(getattr(losses, name).detach().cpu())
            if (
                batch_index % args.gradient_accumulation == 0
                or batch_index == batches
            ):
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0:
                    print(
                        f"PROGRESS epoch={epoch + 1}/{args.epochs} "
                        f"step={global_step} "
                        f"loss={float(losses.total.detach().cpu()):.4f}",
                        flush=True,
                    )
        history.append(
            {
                "epoch": epoch + 1,
                "optimizer_steps": global_step,
                **{
                    name: value / max(1, batches)
                    for name, value in running.items()
                },
            }
        )

    metrics = evaluate(
        model,
        tokenizer,
        dev,
        group_batch_size=args.group_batch_size,
        max_length=args.max_length,
        device=device,
        abstention_threshold=args.abstention_threshold,
        seen_sources=seen_sources,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint"
    model.save_router(checkpoint)
    tokenizer.save_pretrained(checkpoint / "tokenizer")
    report = {
        "status": "complete",
        "method": "multitask_publisher_identity_router_v3",
        "data_contract": "publisher_identity_v3_leakage_controlled",
        "source_url_included": source_url_contract,
        "model": args.model,
        "device": str(device),
        "train_groups": {task: len(rows) for task, rows in train.items()},
        "dev_groups": {task: len(rows) for task, rows in dev.items()},
        "groups_per_task": args.groups_per_task,
        "epochs": args.epochs,
        "seed": args.seed,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "dev": metrics,
        "input_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "checkpoint": str(checkpoint),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
