"""Architecture capability checks."""

from __future__ import annotations

import functools
from typing import Tuple


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    """Return the active CUDA device capability when available."""
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """Return whether the current CUDA device is at least the target SM."""
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def is_sm90_supported() -> bool:
    """Return whether the current device supports SM90 kernels."""
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    """Return whether the current device supports SM100 kernels."""
    return is_arch_supported(10, 0)
