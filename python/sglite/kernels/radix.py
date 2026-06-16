"""Low-level radix-cache comparison helpers."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_radix_module() -> Module:
    """Load the compiled radix comparison extension."""
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """Compare radix-cache key slices efficiently."""
    return _load_radix_module().fast_compare_key(x, y)
