"""RMSNorm layer implementations."""

from typing import Tuple

import torch

from .base import BaseModule


class RMSNorm(BaseModule):
    """RMSNorm layer backed by FlashInfer kernels."""
    def __init__(self, size: int, eps: float) -> None:
        """Allocate RMSNorm weights and bind the kernel implementation."""
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize activations with RMSNorm."""
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """Apply RMSNorm in place."""
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseModule):
    """Fused RMSNorm implementation that delegates to a custom kernel."""
    def __init__(self, size: int, eps: float) -> None:
        """Allocate RMSNorm weights and bind fused add-norm kernels."""
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply standalone RMSNorm or fused residual-add RMSNorm."""
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
