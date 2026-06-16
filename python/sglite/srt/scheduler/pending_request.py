"""Shared scheduler-side request helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglite.sampling_params import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    """Queued request metadata used during scheduling decisions."""
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        """Return the number of prompt tokens waiting for prefill."""
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        """Return the requested maximum decode length."""
        return self.sampling_params.max_tokens
