"""Two-head lightweight router with claim-source interaction features."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression, SGDClassifier  # type: ignore[import-untyped]

from provtimerag.baselines.diagnostic import lexical_scores, tokenize
from provtimerag.routing.lightweight import (
    TEXT_FEATURES,
    FeatureBatch,
    GroupExample,
    balanced_sample_weights,
    numeric_features,
)


def enhanced_text(example: GroupExample, candidate_index: int) -> str:
    group = example.group
    evidence = group.candidates[candidate_index].evidence
    query_terms = [
        term
        for term in dict.fromkeys(
            tokenize(f"{group.query.text} {group.claim.text}")
        )
        if len(term) > 1
    ][:12]
    source_terms = [
        term
        for term in dict.fromkeys(tokenize(evidence.source_id))
        if len(term) > 1 and not term.isdigit()
    ][:4]
    interactions = " ".join(
        f"qsrc_{query}_{source}"
        for query in query_terms
        for source in source_terms
    )
    required_roles = " ".join(group.claim.required_source_roles) or "none"
    return (
        f"query {group.query.text} claim {group.claim.text} "
        f"evidence {evidence.text} source {evidence.source_id} "
        f"source_role {evidence.source_role} required_role {required_roles} "
        f"interactions {interactions}"
    )


def build_multitask_batch(
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
        lexical = lexical_scores(example.group)
        for index, candidate in enumerate(example.group.candidates):
            texts.append(enhanced_text(example, index))
            numbers.append(
                numeric_features(example.group, index, lexical)
            )
            labels.append(int(candidate.gold_route))
            tasks.append(example.task)
            group_ids.append(example.group.group_id)
            evidence_ids.append(candidate.evidence.evidence_id)
    matrix = hstack(
        (
            encoder.transform(texts),
            csr_matrix(np.asarray(numbers, dtype=np.float32)),
        ),
        format="csr",
    )
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


def fit_ranking_head(
    batch: FeatureBatch,
    *,
    random_state: int,
) -> SGDClassifier:
    mask = np.asarray(
        [task != "insufficient" for task in batch.tasks],
        dtype=bool,
    )
    labels = batch.labels[mask]
    tasks = tuple(
        task for task, keep in zip(batch.tasks, mask, strict=True) if keep
    )
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=200,
        tol=1e-4,
        random_state=random_state,
        average=True,
    )
    model.fit(
        batch.matrix[mask],
        labels,
        sample_weight=balanced_sample_weights(labels, tasks),
    )
    return model


def group_decision_matrix(
    examples: Sequence[GroupExample],
    score_map: dict[str, dict[str, float]],
) -> np.ndarray:
    rows: list[list[float]] = []
    for example in examples:
        group = example.group
        ranking_scores = sorted(
            score_map[group.group_id].values(),
            reverse=True,
        )
        lexical = sorted(lexical_scores(group).values(), reverse=True)
        rank_margin = (
            ranking_scores[0] - ranking_scores[1]
            if len(ranking_scores) > 1
            else ranking_scores[0]
        )
        lexical_margin = (
            lexical[0] - lexical[1] if len(lexical) > 1 else lexical[0]
        )
        rows.append(
            [
                ranking_scores[0],
                float(np.mean(ranking_scores)),
                float(np.std(ranking_scores)),
                rank_margin,
                lexical[0],
                lexical_margin,
                math_log_count(len(ranking_scores)),
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def math_log_count(count: int) -> float:
    return float(np.log1p(count))


def fit_abstention_head(
    examples: Sequence[GroupExample],
    score_map: dict[str, dict[str, float]],
    *,
    random_state: int,
) -> LogisticRegression:
    labels = np.asarray(
        [int(example.group.should_abstain) for example in examples],
        dtype=np.int64,
    )
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(group_decision_matrix(examples, score_map), labels)
    return model


def abstention_probabilities(
    model: LogisticRegression,
    examples: Sequence[GroupExample],
    score_map: dict[str, dict[str, float]],
) -> dict[str, float]:
    probabilities = model.predict_proba(
        group_decision_matrix(examples, score_map)
    )
    positive_column = list(model.classes_).index(1)
    return {
        example.group.group_id: float(probability)
        for example, probability in zip(
            examples,
            probabilities[:, positive_column],
            strict=True,
        )
    }


def macro_utility(
    examples: Sequence[GroupExample],
    score_map: dict[str, dict[str, float]],
    abstention: dict[str, float],
    threshold: float,
) -> float:
    hits: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for example in examples:
        group = example.group
        top_id = max(score_map[group.group_id], key=score_map[group.group_id].get)
        predicted_abstain = abstention[group.group_id] >= threshold
        correct = (
            predicted_abstain
            if group.should_abstain
            else (
                not predicted_abstain
                and top_id in group.gold_evidence_ids
            )
        )
        hits[example.task] += int(correct)
        totals[example.task] += 1
    return sum(hits[task] / totals[task] for task in totals) / len(totals)


def tune_abstention_threshold(
    examples: Sequence[GroupExample],
    score_map: dict[str, dict[str, float]],
    abstention: dict[str, float],
) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    utilities = [
        macro_utility(
            examples,
            score_map,
            abstention,
            float(threshold),
        )
        for threshold in thresholds
    ]
    index = int(np.argmax(utilities))
    return float(thresholds[index]), float(utilities[index])
