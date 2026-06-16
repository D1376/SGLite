"""Base interfaces for MoE backends."""

from abc import ABC, abstractmethod

import torch


class BaseMoeBackend(ABC):
    """Base class for MoE backends."""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        """Apply the MoE kernel to the provided hidden states and expert weights."""
        ...
