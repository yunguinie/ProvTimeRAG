"""Transformer backbone wrapper for the multi-task Provenance-State Router."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel  # type: ignore[import-untyped]

from provtimerag.routing.multitask import ProvenanceRouterHeads, RouterOutputs


class MultiTaskProvenanceRouter(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = ProvenanceRouterHeads(hidden_size, dropout=dropout)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> "MultiTaskProvenanceRouter":
        backbone = AutoModel.from_pretrained(str(model_name_or_path))
        hidden_size = int(backbone.config.hidden_size)
        if gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        router = cls(backbone, hidden_size, dropout=dropout)
        weight_path = Path(model_name_or_path) / "model.safetensors"
        if weight_path.is_file():
            from safetensors import safe_open

            keys = (
                "classifier.dense.weight",
                "classifier.dense.bias",
                "classifier.out_proj.weight",
                "classifier.out_proj.bias",
            )
            with safe_open(str(weight_path), framework="pt", device="cpu") as source:
                if all(key in source.keys() for key in keys):
                    router.heads.initialize_route_head(
                        *(source.get_tensor(key) for key in keys)
                    )
        return router

    def encode(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs: Any = self.backbone(**encoded)
        if not hasattr(outputs, "last_hidden_state"):
            raise ValueError("backbone must return last_hidden_state")
        return outputs.last_hidden_state[:, 0]

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        group_index: torch.Tensor,
        num_groups: int,
    ) -> RouterOutputs:
        embeddings = self.encode(encoded)
        return self.heads(embeddings, group_index, num_groups)

    def save_router(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        backbone_path = output / "backbone"
        if not hasattr(self.backbone, "save_pretrained"):
            raise ValueError("backbone does not support save_pretrained")
        self.backbone.save_pretrained(backbone_path)
        torch.save(self.heads.state_dict(), output / "router_heads.pt")
