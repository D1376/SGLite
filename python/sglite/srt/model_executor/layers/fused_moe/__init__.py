"""Public exports for MoE."""

from __future__ import annotations

from typing import Protocol

from sglite.srt.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    """Callable protocol used to register MoE backend builders."""

    def __call__(self) -> BaseMoeBackend:
        """Build the MoE backend implementation."""
        ...


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    """Create the fused Triton MoE backend."""
    from .fused import FusedMoe

    return FusedMoe()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    """Create a registered MoE backend by name."""
    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
]
