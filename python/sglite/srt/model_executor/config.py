"""Engine configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from sglite.srt.distributed import DistributedInfo
from sglite.srt.utils import cached_load_hf_config

if TYPE_CHECKING:
    from sglite.srt.model_executor.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    """Configuration required to construct one model executor worker."""
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # Overrides automatic KV-cache page sizing.

    @cached_property
    def hf_config(self):
        """Load and cache the Hugging Face model configuration."""
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        """Convert Hugging Face metadata into SGLite model metadata."""
        from sglite.srt.model_executor.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config, model_path=self.model_path)

    @property
    def max_seq_len(self) -> int:
        """Return the effective maximum sequence length."""
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        """Return the maximum token count accepted by one forward pass."""
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        """Return the default torch.distributed endpoint."""
        return "tcp://127.0.0.1:2333"
