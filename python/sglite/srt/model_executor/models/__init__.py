"""Public exports for models."""

from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .registry import get_model_class
from sglite.srt.model_executor.model_loader import load_weight


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    """Instantiate the model class declared by the Hugging Face config."""
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
