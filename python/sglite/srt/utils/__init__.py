"""Public exports for utilities."""

from .arch import is_arch_supported, is_sm90_supported, is_sm100_supported
from .logger import configure_external_loggers, dim_text, init_logger, print_banner, should_use_color
from .misc import UNSET, Unset, align_ceil, align_down, div_ceil, div_even
from .registry import Registry
from .torch_utils import nvtx_annotate, torch_dtype


def __getattr__(name: str):
    """Lazily load helpers that depend on optional serving/model-loading packages."""
    if name in ("cached_load_hf_config", "download_hf_weight", "load_tokenizer"):
        from . import hf

        return getattr(hf, name)
    if name in (
        "ZmqAsyncPullQueue",
        "ZmqAsyncPushQueue",
        "ZmqPubQueue",
        "ZmqPullQueue",
        "ZmqPushQueue",
        "ZmqSubQueue",
    ):
        from . import zmq_queue

        return getattr(zmq_queue, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "cached_load_hf_config",
    "configure_external_loggers",
    "dim_text",
    "download_hf_weight",
    "load_tokenizer",
    "init_logger",
    "print_banner",
    "is_arch_supported",
    "is_sm90_supported",
    "is_sm100_supported",
    "should_use_color",
    "div_even",
    "div_ceil",
    "align_ceil",
    "align_down",
    "UNSET",
    "Unset",
    "torch_dtype",
    "nvtx_annotate",
    "Registry",
    "ZmqPushQueue",
    "ZmqPullQueue",
    "ZmqPubQueue",
    "ZmqSubQueue",
    "ZmqAsyncPushQueue",
    "ZmqAsyncPullQueue",
]
