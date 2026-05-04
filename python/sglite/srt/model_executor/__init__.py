"""Public exports for model execution."""

__all__ = ["Engine", "EngineConfig", "ForwardOutput", "BatchSamplingArgs"]


def __getattr__(name: str):
    """Lazily expose model-executor entry points without importing heavy runtime dependencies."""
    if name == "EngineConfig":
        from .config import EngineConfig

        return EngineConfig
    if name in ("Engine", "ForwardOutput"):
        from .engine import Engine, ForwardOutput

        return {"Engine": Engine, "ForwardOutput": ForwardOutput}[name]
    if name == "BatchSamplingArgs":
        from .sampler import BatchSamplingArgs

        return BatchSamplingArgs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
