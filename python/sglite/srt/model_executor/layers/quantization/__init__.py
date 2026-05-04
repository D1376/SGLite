"""Quantization module for SGLite.

Supports:
- AWQ (Activation-aware Weight Quantization) using Triton kernels (SM75+)
- AWQ Marlin using Marlin CUDA kernels (SM80+)
"""

# Only import pure-Python base types eagerly. AWQ Triton and Marlin backends
# can initialize CUDA or require optional kernel packages, so defer those
# imports until the engine selects a quantization path.
from .base_config import LinearMethodBase, QuantizationConfig, set_weight_attrs
from .unquant import UnquantizedLinearMethod


def get_quantization_config(method_name: str) -> type:
    """Return the quantization config class for a method name."""
    method_name = method_name.lower()
    if method_name == "awq":
        from .awq import AWQConfig

        return AWQConfig
    if method_name == "awq_marlin":
        from .awq_marlin import AWQMarlinConfig

        return AWQMarlinConfig
    raise ValueError(
        f"Unknown quantization method: {method_name!r}. "
        f"Supported methods: ['awq', 'awq_marlin']"
    )


def __getattr__(name: str):
    """Lazily expose heavy quantization classes without importing CUDA early."""
    if name in ("AWQConfig", "AWQLinearMethod"):
        from . import awq

        return getattr(awq, name)
    if name in ("AWQMarlinConfig", "AWQMarlinLinearMethod"):
        from . import awq_marlin

        return getattr(awq_marlin, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AWQConfig",
    "AWQLinearMethod",
    "AWQMarlinConfig",
    "AWQMarlinLinearMethod",
    "LinearMethodBase",
    "QuantizationConfig",
    "UnquantizedLinearMethod",
    "set_weight_attrs",
    "get_quantization_config",
]
