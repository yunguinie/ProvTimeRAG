"""Benchmark publisher rerankers with one frozen, inference-only contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def timed_repeats(
    function: Callable[[], None], *, repeats: int, device: Any
) -> tuple[list[float], float | None]:
    import torch

    seconds: list[float] = []
    peaks: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peaks.append(torch.cuda.max_memory_allocated(device) / (1024**2))
        seconds.append(time.perf_counter() - started)
    return seconds, max(peaks) if peaks else None


def load_c2(args: argparse.Namespace, device: Any, dtype: Any):
    import torch
    from transformers import AutoTokenizer

    from provtimerag.routing.multitask_data import TASK_SOURCE, flatten_batch
    from provtimerag.routing.multitask_model import MultiTaskProvenanceRouter
    from scripts.train import train_multitask_provenance_router as legacy
    from scripts.train.train_multitask_publisher_router_v3 import load_task

    examples = load_task(args.input, TASK_SOURCE, None)
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

    def run(selected: list[Any]) -> None:
        with torch.inference_mode():
            for start in range(0, len(selected), args.group_batch_size):
                batch = selected[start : start + args.group_batch_size]
                requests, candidates, targets = flatten_batch(batch, device)
                encoded = legacy.encode(
                    tokenizer,
                    requests,
                    candidates,
                    max_length=args.max_length,
                    device=device,
                )
                model(encoded, targets.group_index, len(batch))

    candidates = sum(len(example.candidate_texts) for example in examples)
    return model, examples, candidates, lambda: run(examples), lambda: run(
        examples[: args.warmup_groups]
    )


def load_classifier(args: argparse.Namespace, device: Any, dtype: Any):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from provtimerag.data import read_bundles_jsonl
    from scripts.evaluate.run_c3_strong_cross_encoder_baseline_v1 import observable_pair

    bundles = list(read_bundles_jsonl(args.input))
    groups = [group for bundle in bundles for group in bundle.groups]
    pairs = [
        observable_pair(group, index, input_view="publisher_visible")
        for group in groups
        for index in range(len(group.candidates))
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()

    def run(selected: list[tuple[str, str]]) -> None:
        with torch.inference_mode():
            for start in range(0, len(selected), args.candidate_batch_size):
                batch = selected[start : start + args.candidate_batch_size]
                encoded = tokenizer(
                    [query for query, _ in batch],
                    [document for _, document in batch],
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded, return_dict=True)

    warmup_candidates = sum(
        len(group.candidates) for group in groups[: args.warmup_groups]
    )
    return model, groups, len(pairs), lambda: run(pairs), lambda: run(
        pairs[:warmup_candidates]
    )


def load_qwen3(args: argparse.Namespace, device: Any, dtype: Any):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from provtimerag.data import read_bundles_jsonl
    from scripts.evaluate.run_c3_qwen3_reranker_baseline_v1 import (
        PREFIX,
        SUFFIX,
        model_text,
    )

    bundles = list(read_bundles_jsonl(args.input))
    groups = [group for bundle in bundles for group in bundle.groups]
    texts = [
        model_text(group, index, "publisher_visible")
        for group in groups
        for index in range(len(group.candidates))
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = args.max_length - len(prefix) - len(suffix)
    if budget <= 0:
        raise ValueError("max length is shorter than the Qwen3 wrapper")

    def run(selected: list[str]) -> None:
        with torch.inference_mode():
            for start in range(0, len(selected), args.candidate_batch_size):
                batch = tokenizer(
                    selected[start : start + args.candidate_batch_size],
                    padding=False,
                    truncation=True,
                    max_length=budget,
                    return_attention_mask=False,
                )["input_ids"]
                wrapped = [prefix + ids + suffix for ids in batch]
                encoded = tokenizer.pad(
                    {"input_ids": wrapped}, padding=True, return_tensors="pt"
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model(**encoded, return_dict=True)

    warmup_candidates = sum(
        len(group.candidates) for group in groups[: args.warmup_groups]
    )
    return model, groups, len(texts), lambda: run(texts), lambda: run(
        texts[:warmup_candidates]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("c2", "classifier", "qwen3"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--group-batch-size", type=int, default=2)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--warmup-groups", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2 or args.warmup_groups < 1:
        raise ValueError("repeats must be >=2 and warmup-groups must be positive")
    if args.kind == "c2" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for c2")
    if args.kind != "c2" and not args.model:
        raise ValueError("--model is required for rerankers")

    import torch

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32
    loader = {"c2": load_c2, "classifier": load_classifier, "qwen3": load_qwen3}[
        args.kind
    ]
    load_started = time.perf_counter()
    model, groups, candidates, run, warmup = loader(args, device, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - load_started
    warmup()
    seconds, peak = timed_repeats(run, repeats=args.repeats, device=device)
    median = statistics.median(seconds)
    artifact = args.checkpoint if args.kind == "c2" else Path(args.model)
    report = {
        "status": "complete",
        "method": "publisher_efficiency_benchmark_v1",
        "contract": "same_input_warmup_inference_only_no_accuracy_selection",
        "kind": args.kind,
        "artifact": str(artifact),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(dtype),
        "max_length": args.max_length,
        "group_batch_size": args.group_batch_size if args.kind == "c2" else None,
        "candidate_batch_size": args.candidate_batch_size if args.kind != "c2" else None,
        "warmup_groups": min(args.warmup_groups, len(groups)),
        "repeats": args.repeats,
        "groups": len(groups),
        "candidates": candidates,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "artifact_bytes": artifact_bytes(artifact),
        "model_load_seconds": model_load_seconds,
        "inference_seconds": seconds,
        "median_inference_seconds": median,
        "groups_per_second": len(groups) / median,
        "candidates_per_second": candidates / median,
        "median_milliseconds_per_group": 1000 * median / len(groups),
        "peak_cuda_memory_mb": peak,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
