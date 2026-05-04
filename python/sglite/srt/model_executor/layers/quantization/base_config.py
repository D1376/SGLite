"""Base types for quantized linear methods."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
) -> None:
    """Attach loader metadata to a weight tensor without overwriting fields."""
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        if hasattr(weight, key):
            raise RuntimeError(f"Overwriting existing attribute {key} on weight tensor.")
        setattr(weight, key, value)


class LinearMethodBase(ABC):
    """Base class for quantized linear-method implementations."""

    @abstractmethod
    def create_weights(self, input_size: int, output_size: int) -> Dict[str, Any]:
        """Allocate tensors and metadata required by this linear method."""
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the loaded quantized weights to the input activations."""
        raise NotImplementedError

    def process_weights_after_loading(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Optionally repack or augment weights after checkpoint loading."""
        return weights


class QuantizationConfig(ABC):
    """Configuration container for a quantization scheme."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the quantization method name used in model metadata."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """Return activation dtypes supported by this method."""
        raise NotImplementedError

    @abstractmethod
    def get_min_capability(self) -> int:
        """Return the minimum CUDA compute capability required."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> List[str]:
        """Return quantization config filenames recognized by this method."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        """Build the configuration object from model or runtime metadata."""
        raise NotImplementedError

    @staticmethod
    def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:
        """Return the first value present under any accepted metadata key."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(f"Cannot find any of {keys} in the model's quantization config.")

    @abstractmethod
    def get_linear_method(self) -> LinearMethodBase:
        """Create the linear method that consumes this config."""
        raise NotImplementedError

    @abstractmethod
    def get_scaled_act_names(self) -> List[str]:
        """Return activation names that require AWQ activation scaling."""
        raise NotImplementedError
