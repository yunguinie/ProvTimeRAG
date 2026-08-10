"""Frozen data adapters and balanced sampling for the multi-task Router."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

import torch

from provtimerag.data import ClaimBundle, ClaimRouteGroup, EvidenceRecord
from provtimerag.routing.multitask import RouterTargets


TASK_SOURCE = "source"
TASK_TEMPORAL = "temporal_version"
TASK_INSUFFICIENT = "insufficient"
TASKS = (TASK_SOURCE, TASK_TEMPORAL, TASK_INSUFFICIENT)


@dataclass(frozen=True)
class RouterGroupExample:
    group_id: str
    task: str
    request_text: str
    candidate_texts: tuple[str, ...]
    candidate_source_ids: tuple[str, ...]
    gold_route: tuple[bool, ...]
    source_match: tuple[bool, ...]
    temporal_valid: tuple[bool, ...]
    version_valid: tuple[bool, ...]
    should_abstain: bool


def request_text(group: ClaimRouteGroup) -> str:
    roles = ", ".join(group.claim.required_source_roles) or "unspecified"
    target_time = group.claim.target_time.isoformat() if group.claim.target_time else "unspecified"
    return (
        f"Question: {group.query.text}\n"
        f"Atomic claim need: {group.claim.text}\n"
        f"Required source roles: {roles}\n"
        f"Temporal scope: {group.claim.temporal_scope}\n"
        f"Target time: {target_time}"
    )


def candidate_text(evidence: EvidenceRecord, *, include_source_url: bool = False) -> str:
    metadata = evidence.metadata
    source_url = str(metadata.get("source_url") or metadata.get("cached_source_url") or "")
    source_domain = urlparse(source_url).netloc if source_url else ""
    fields = [
        f"Evidence: {evidence.text}",
        f"Source identity: {evidence.source_id}",
        f"Source role: {evidence.source_role}",
        f"Source medium: {metadata.get('source_medium') or 'unspecified'}",
    ]
    if include_source_url:
        fields.extend(
            [
                f"Source URL: {source_url or 'unspecified'}",
                f"Source domain: {source_domain or 'unspecified'}",
            ]
        )
    fields.extend(
        [
            f"Document title: {metadata.get('document_title') or 'unspecified'}",
            f"Publication time: {evidence.publication_time.isoformat() if evidence.publication_time else 'unspecified'}",
            f"Valid from: {evidence.valid_from.isoformat() if evidence.valid_from else 'unspecified'}",
            f"Valid to: {evidence.valid_to.isoformat() if evidence.valid_to else 'open'}",
            f"Version: {evidence.version_id}",
        ]
    )
    return "\n".join(fields)


def group_example(
    group: ClaimRouteGroup,
    task: str,
    *,
    include_source_url: bool = False,
) -> RouterGroupExample:
    if task not in TASKS:
        raise ValueError(f"unknown Router task: {task}")
    if not group.candidates:
        raise ValueError("Router groups must contain candidates")
    return RouterGroupExample(
        group_id=group.group_id,
        task=task,
        request_text=request_text(group),
        candidate_texts=tuple(
            candidate_text(candidate.evidence, include_source_url=include_source_url)
            for candidate in group.candidates
        ),
        candidate_source_ids=tuple(
            candidate.evidence.source_id for candidate in group.candidates
        ),
        gold_route=tuple(candidate.gold_route for candidate in group.candidates),
        source_match=tuple(candidate.source_match for candidate in group.candidates),
        temporal_valid=tuple(candidate.temporal_valid for candidate in group.candidates),
        version_valid=tuple(candidate.version_valid for candidate in group.candidates),
        should_abstain=group.should_abstain,
    )


def bundle_examples(
    bundles: Iterable[ClaimBundle],
    task: str,
    *,
    include_source_url: bool = False,
) -> list[RouterGroupExample]:
    return [
        group_example(group, task, include_source_url=include_source_url)
        for bundle in bundles
        for group in bundle.groups
    ]


def deterministic_order(examples: Iterable[RouterGroupExample], seed: int) -> list[RouterGroupExample]:
    return sorted(
        examples,
        key=lambda example: hashlib.sha256(
            f"{seed}:{example.group_id}".encode("utf-8")
        ).hexdigest(),
    )


def balanced_epoch(
    examples_by_task: dict[str, list[RouterGroupExample]],
    *,
    groups_per_task: int,
    seed: int,
) -> list[RouterGroupExample]:
    if groups_per_task <= 0:
        raise ValueError("groups_per_task must be positive")
    missing = [task for task in TASKS if not examples_by_task.get(task)]
    if missing:
        raise ValueError(f"missing examples for tasks: {missing}")
    randomizer = random.Random(seed)
    selected: dict[str, list[RouterGroupExample]] = {}
    for task in TASKS:
        ordered = deterministic_order(examples_by_task[task], seed)
        if len(ordered) >= groups_per_task:
            selected[task] = ordered[:groups_per_task]
        else:
            selected[task] = [ordered[index % len(ordered)] for index in range(groups_per_task)]
    result: list[RouterGroupExample] = []
    for index in range(groups_per_task):
        task_order = list(TASKS)
        randomizer.shuffle(task_order)
        result.extend(selected[task][index] for task in task_order)
    return result


def flatten_batch(
    examples: list[RouterGroupExample], device: torch.device | None = None
) -> tuple[list[str], list[str], RouterTargets]:
    if not examples:
        raise ValueError("cannot flatten an empty Router batch")
    requests: list[str] = []
    candidates: list[str] = []
    group_index: list[int] = []
    gold_route: list[bool] = []
    source_match: list[bool] = []
    temporal_valid: list[bool] = []
    version_valid: list[bool] = []
    source_mask: list[bool] = []
    temporal_mask: list[bool] = []
    version_mask: list[bool] = []
    for group, example in enumerate(examples):
        count = len(example.candidate_texts)
        requests.extend([example.request_text] * count)
        candidates.extend(example.candidate_texts)
        group_index.extend([group] * count)
        gold_route.extend(example.gold_route)
        source_match.extend(example.source_match)
        temporal_valid.extend(example.temporal_valid)
        version_valid.extend(example.version_valid)
        source_mask.extend([example.task == TASK_SOURCE] * count)
        temporal_mask.extend([example.task == TASK_TEMPORAL] * count)
        version_mask.extend([example.task == TASK_TEMPORAL] * count)
    tensor_device = device or torch.device("cpu")
    targets = RouterTargets(
        group_index=torch.tensor(group_index, dtype=torch.long, device=tensor_device),
        gold_route=torch.tensor(gold_route, dtype=torch.bool, device=tensor_device),
        source_match=torch.tensor(source_match, dtype=torch.float, device=tensor_device),
        temporal_valid=torch.tensor(temporal_valid, dtype=torch.float, device=tensor_device),
        version_valid=torch.tensor(version_valid, dtype=torch.float, device=tensor_device),
        should_abstain=torch.tensor(
            [example.should_abstain for example in examples], dtype=torch.float, device=tensor_device
        ),
        source_mask=torch.tensor(source_mask, dtype=torch.bool, device=tensor_device),
        temporal_mask=torch.tensor(temporal_mask, dtype=torch.bool, device=tensor_device),
        version_mask=torch.tensor(version_mask, dtype=torch.bool, device=tensor_device),
    )
    return requests, candidates, targets
