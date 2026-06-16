"""Base model abstractions for decoder-only architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from sglite.srt.model_executor.layers import BaseModule

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseModule):
    """Base class for decoder-only language models."""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Run the model for the current runtime batch and return the output logits."""
        ...
