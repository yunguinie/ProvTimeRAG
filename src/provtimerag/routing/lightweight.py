"""Lightweight learnability gate for provenance-state routing."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import SGDClassifier  # type: ignore[import-untyped]

from provtimerag.baselines.diagnostic import lexical_scores
from provtimerag.data import ClaimRouteGroup

TEXT_FEATURES = 2**18
TASK_NAMES = ("temporal", "source", "insufficient")


@dataclass(frozen=True)
class GroupExample:
    group: ClaimRouteGroup
    task: str


@dataclass(frozen=True)
class FeatureBatch:
    matrix: Any
    labels: np.ndarray
    tasks: tuple[str, ...]
    group_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


def observable_text(group: ClaimRouteGroup, candidate_index: int) -> str:
    evidence = group.candidates[candidate_index].evidence
    required_roles = " ".join(group.claim.required_source_roles) or "none"
    return (
        f"query {group.query.text} claim {group.claim.text} "
        f"evidence {evidence.text} source {evidence.source_id} "
        f"source_role {evidence.source_role} required_role {required_roles}"
    )


def temporal_compatibility(
    target: datetime | None,
    valid_from: datetime | None,
    valid_to: datetime | None,
) -> float:
    if target is None or (valid_from is None and valid_to is None):
        return 0.0
    if valid_from is not None and target < valid_from:
        return -1.0
    if valid_to is not None and target > valid_to:
        return -1.0
    return 1.0


def numeric_features(
    group: ClaimRouteGroup,
    candidate_index: int,
    lexical: dict[str, float],
) -> list[float]:
    candidate = group.candidates[candidate_index]
    evidence = candidate.evidence
    publication_times = [
        item.evidence.publication_time
        for item in group.candidates
        if item.evidence.publication_time is not None
    ]
    newest = max(publication_times) if publication_times else None
    target = group.claim.target_time
    age_days = 0.0
    if target is not None and evidence.publication_time is not None:
        age_days = max(
            -10.0,
            min(10.0, (target - evidence.publication_time).days / 365.0),
        )
    required = set(group.claim.required_source_roles)
    return [
        lexical[evidence.evidence_id],
        temporal_compatibility(target, evidence.valid_from, evidence.valid_to),
        float(newest is not None and evidence.publication_time == newest),
        float(bool(required) and evidence.source_role in required),
        float(bool(evidence.supersedes_ids)),
        float(evidence.publication_time is not None),
        float(evidence.valid_from is not None or evidence.valid_to is not None),
        age_days,
        math.log1p(len(evidence.text)),
    ]


def build_feature_batch(
    examples: Sequence[GroupExample],
    vectorizer: HashingVectorizer | None = None,
) -> tuple[FeatureBatch, HashingVectorizer]:
    encoder = vectorizer or HashingVectorizer(
        n_features=TEXT_FEATURES,
        alternate_sign=False,
        ngram_range=(1, 2),
        norm="l2",
        lowercase=True,
    )
    texts: list[str] = []
    numbers: list[list[float]] = []
    labels: list[int] = []
    tasks: list[str] = []
    group_ids: list[str] = []
    evidence_ids: list[str] = []
    for example in examples:
        if example.task not in TASK_NAMES:
            raise ValueError(f"unknown task: {example.task}")
        lexical = lexical_scores(example.group)
        for index, candidate in enumerate(example.group.candidates):
            texts.append(observable_text(example.group, index))
            numbers.append(numeric_features(example.group, index, lexical))
            labels.append(int(candidate.gold_route))
            tasks.append(example.task)
            group_ids.append(example.group.group_id)
            evidence_ids.append(candidate.evidence.evidence_id)
    text_matrix = encoder.transform(texts)
    numeric_matrix = csr_matrix(np.asarray(numbers, dtype=np.float32))
    matrix = hstack((text_matrix, numeric_matrix), format="csr")
    return (
        FeatureBatch(
            matrix=matrix,
            labels=np.asarray(labels, dtype=np.int64),
            tasks=tuple(tasks),
            group_ids=tuple(group_ids),
            evidence_ids=tuple(evidence_ids),
        ),
        encoder,
    )


def balanced_sample_weights(
    labels: np.ndarray,
    tasks: Sequence[str],
) -> np.ndarray:
    buckets = Counter(zip(tasks, labels.tolist(), strict=True))
    labels_per_task = {
        task: len(
            {
                label
                for observed_task, label in buckets
                if observed_task == task
            }
        )
        for task in set(tasks)
    }
    weights = np.asarray(
        [
            1.0
            / (
                buckets[(task, int(label))] * labels_per_task[task]
            )
            for task, label in zip(tasks, labels, strict=True)
        ],
        dtype=np.float64,
    )
    return weights * (len(weights) / weights.sum())


def fit_router(
    batch: FeatureBatch,
    *,
    random_state: int = 42,
) -> SGDClassifier:
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=50,
        tol=1e-4,
        random_state=random_state,
        average=True,
    )
    classifier.fit(
        batch.matrix,
        batch.labels,
        sample_weight=balanced_sample_weights(batch.labels, batch.tasks),
    )
    return classifier


def positive_scores(
    classifier: SGDClassifier,
    batch: FeatureBatch,
) -> np.ndarray:
    probabilities = classifier.predict_proba(batch.matrix)
    positive_column = list(classifier.classes_).index(1)
    return np.asarray(probabilities[:, positive_column], dtype=np.float64)


def grouped_scores(
    batch: FeatureBatch,
    scores: Sequence[float],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for group_id, evidence_id, score in zip(
        batch.group_ids,
        batch.evidence_ids,
        scores,
        strict=True,
    ):
        output.setdefault(group_id, {})[evidence_id] = float(score)
    return output


def threshold_macro_utility(
    examples: Iterable[GroupExample],
    scores: dict[str, dict[str, float]],
    threshold: float,
) -> float:
    hits: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for example in examples:
        group = example.group
        group_scores = scores[group.group_id]
        top_id = max(group_scores, key=group_scores.get)
        abstain = group_scores[top_id] < threshold
        correct = (
            abstain
            if group.should_abstain
            else (not abstain and top_id in group.gold_evidence_ids)
        )
        hits[example.task] += int(correct)
        totals[example.task] += 1
    task_scores = [
        hits[task] / totals[task] for task in TASK_NAMES if totals[task]
    ]
    return sum(task_scores) / len(task_scores)


def tune_threshold(
    examples: Sequence[GroupExample],
    scores: dict[str, dict[str, float]],
) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 91)
    utilities = [
        threshold_macro_utility(examples, scores, float(value))
        for value in candidates
    ]
    best_index = int(np.argmax(utilities))
    return float(candidates[best_index]), float(utilities[best_index])
