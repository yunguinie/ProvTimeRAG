"""Coverage-aware wrapper for the bounded listwise C3 evaluator."""

from __future__ import annotations

import numpy as np
import torch

from scripts.evaluate import run_c3_structured_source_swap_fixed as bounded


implementation = bounded.implementation
MULTI_SOURCE_LOSS_WEIGHT = 0.50


def fit_assignment_head_coverage(
    bundles,
    scores,
    candidate_limit: int,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    weight = torch.zeros(len(implementation.FEATURE_NAMES), requires_grad=True)
    optimizer = torch.optim.Adam([weight], lr=learning_rate)
    for _ in range(epochs):
        for bundle in bundles:
            features, gold, _ = implementation.enumerate_bundle(bundle, scores, candidate_limit)
            x = torch.tensor(features, dtype=torch.float32)
            logits = x @ weight
            target = torch.tensor(gold, dtype=torch.bool)
            if not bool(target.any()):
                continue
            assignment_loss = -torch.logsumexp(logits[target], dim=0) + torch.logsumexp(logits, dim=0)
            multi_mask = torch.tensor(features[:, 2] > 0, dtype=torch.bool)
            multi_probability = torch.softmax(logits, dim=0)[multi_mask].sum().clamp(1e-6, 1 - 1e-6)
            gold_sources = {
                candidate.evidence.source_id
                for group in bundle.groups
                for candidate in group.candidates
                if candidate.gold_route
            }
            gold_multi = float(len(gold_sources) > 1)
            coverage_loss = -(
                gold_multi * torch.log(multi_probability)
                + (1.0 - gold_multi) * torch.log(1.0 - multi_probability)
            )
            loss = assignment_loss + MULTI_SOURCE_LOSS_WEIGHT * coverage_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return weight.detach().numpy()


implementation.fit_assignment_head = fit_assignment_head_coverage


if __name__ == "__main__":
    implementation.main()
