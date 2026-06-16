"""Torch-specific context helpers."""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    """Temporarily switch the default torch dtype."""
    import torch  # Defer the real import until runtime.

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    """Annotate a code region with NVTX markers when available."""
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        """Wrap a callable so its execution is labeled with an NVTX range."""

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            """Emit an NVTX range around the wrapped function call."""
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
