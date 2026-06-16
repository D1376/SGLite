"""Sampling parameter definitions for generation requests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Sampling controls for a generation request."""
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        """Return whether these settings reduce sampling to argmax."""
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
