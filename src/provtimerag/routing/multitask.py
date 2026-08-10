"""Neural heads and losses for the multi-task Provenance-State Router."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RouterOutputs:
    route_logits: torch.Tensor
    source_logits: torch.Tensor
    temporal_logits: torch.Tensor
    version_logits: torch.Tensor
    abstention_logits: torch.Tensor


@dataclass(frozen=True)
class RouterTargets:
    group_index: torch.Tensor
    gold_route: torch.Tensor
    source_match: torch.Tensor
    temporal_valid: torch.Tensor
    version_valid: torch.Tensor
    should_abstain: torch.Tensor
    source_mask: torch.Tensor
    temporal_mask: torch.Tensor
    version_mask: torch.Tensor


@dataclass(frozen=True)
class RouterLoss:
    total: torch.Tensor
    route: torch.Tensor
    source: torch.Tensor
    temporal: torch.Tensor
    version: torch.Tensor
    abstention: torch.Tensor


def _group_mean_max(
    embeddings: torch.Tensor, group_index: torch.Tensor, num_groups: int
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError("candidate embeddings must have shape [candidates, hidden]")
    if group_index.ndim != 1 or group_index.numel() != embeddings.shape[0]:
        raise ValueError("group_index must align with candidate embeddings")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    hidden = embeddings.shape[1]
    sums = embeddings.new_zeros((num_groups, hidden))
    counts = embeddings.new_zeros((num_groups, 1))
    sums.index_add_(0, group_index, embeddings)
    counts.index_add_(0, group_index, embeddings.new_ones((embeddings.shape[0], 1)))
    if torch.any(counts == 0):
        raise ValueError("every group must contain at least one candidate")
    means = sums / counts
    maxima = torch.stack(
        [embeddings[group_index == group].max(dim=0).values for group in range(num_groups)]
    )
    return torch.cat((means, maxima), dim=-1)


class ProvenanceRouterHeads(nn.Module):
    """Candidate-level task heads plus a permutation-invariant group abstention head."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.dropout = nn.Dropout(dropout)
        self.candidate_heads = nn.Linear(hidden_size, 4)
        self.route_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.abstention_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        candidate_embeddings: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int,
    ) -> RouterOutputs:
        candidate_logits = self.candidate_heads(self.dropout(candidate_embeddings))
        route_logits = self.route_head(candidate_embeddings).squeeze(-1)
        pooled = _group_mean_max(candidate_embeddings, group_index, num_groups)
        abstention = self.abstention_head(pooled).squeeze(-1)
        return RouterOutputs(
            route_logits=route_logits,
            source_logits=candidate_logits[:, 1],
            temporal_logits=candidate_logits[:, 2],
            version_logits=candidate_logits[:, 3],
            abstention_logits=abstention,
        )

    def initialize_route_head(
        self,
        dense_weight: torch.Tensor,
        dense_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ) -> None:
        expected = self.route_head[0].weight.shape
        if dense_weight.shape != expected:
            raise ValueError(
                f"classifier dense weight shape {tuple(dense_weight.shape)} != {tuple(expected)}"
            )
        if out_proj_weight.shape != self.route_head[2].weight.shape:
            raise ValueError("classifier output projection shape does not match route head")
        with torch.no_grad():
            self.route_head[0].weight.copy_(dense_weight)
            self.route_head[0].bias.copy_(dense_bias)
            self.route_head[2].weight.copy_(out_proj_weight)
            self.route_head[2].bias.copy_(out_proj_bias)


def _masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask.bool()
    if logits.shape != labels.shape or logits.shape != mask.shape:
        raise ValueError("logits, labels, and mask must have equal shapes")
    if not torch.any(active):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[active], labels[active].float())


def group_pairwise_route_loss(
    route_logits: torch.Tensor,
    gold_route: torch.Tensor,
    group_index: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    """Average all positive-negative comparisons within answerable groups."""
    losses: list[torch.Tensor] = []
    for group in range(num_groups):
        selected = group_index == group
        positives = route_logits[selected & gold_route.bool()]
        negatives = route_logits[selected & ~gold_route.bool()]
        if positives.numel() and negatives.numel():
            differences = positives[:, None] - negatives[None, :]
            losses.append(-F.logsigmoid(differences).mean())
    if not losses:
        return route_logits.sum() * 0.0
    return torch.stack(losses).mean()


def multitask_router_loss(
    outputs: RouterOutputs,
    targets: RouterTargets,
    *,
    route_weight: float = 1.0,
    source_weight: float = 1.0,
    temporal_weight: float = 1.0,
    version_weight: float = 1.0,
    abstention_weight: float = 1.0,
) -> RouterLoss:
    num_groups = targets.should_abstain.numel()
    if outputs.abstention_logits.numel() != num_groups:
        raise ValueError("abstention logits must align with group targets")
    route = group_pairwise_route_loss(
        outputs.route_logits, targets.gold_route, targets.group_index, num_groups
    )
    source = _masked_bce(outputs.source_logits, targets.source_match, targets.source_mask)
    temporal = _masked_bce(
        outputs.temporal_logits, targets.temporal_valid, targets.temporal_mask
    )
    version = _masked_bce(outputs.version_logits, targets.version_valid, targets.version_mask)
    abstention = F.binary_cross_entropy_with_logits(
        outputs.abstention_logits, targets.should_abstain.float()
    )
    total = (
        route_weight * route
        + source_weight * source
        + temporal_weight * temporal
        + version_weight * version
        + abstention_weight * abstention
    )
    return RouterLoss(
        total=total,
        route=route,
        source=source,
        temporal=temporal,
        version=version,
        abstention=abstention,
    )
