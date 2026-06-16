"""AWQ quantization configuration and linear methods."""

from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from .base_config import LinearMethodBase, QuantizationConfig, set_weight_attrs


class AWQConfig(QuantizationConfig):
    """Configuration container for AWQ."""
    def __init__(self, weight_bits: int, group_size: int, zero_point: bool) -> None:
        """Validate AWQ bit width and derive packing metadata."""
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point

        if self.weight_bits != 4:
            raise ValueError(
                f"Currently, only 4-bit weight quantization is supported for AWQ, "
                f"but got {self.weight_bits} bits."
            )
        self.pack_factor = 32 // self.weight_bits

    def __repr__(self) -> str:
        """Return a debug representation of the AWQ config."""
        return (
            f"AWQConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point})"
        )

    def get_name(self) -> str:
        """Return the AWQ method name used in model metadata."""
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """Return activation dtypes supported by AWQ Triton."""
        return [torch.half]

    def get_min_capability(self) -> int:
        """Return the minimum CUDA compute capability required by AWQ."""
        return 75

    @staticmethod
    def get_config_filenames() -> List[str]:
        """Return quantization config filenames recognized for AWQ."""
        return ["quant_config.json", "quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQConfig":
        """Build the configuration object from model or runtime metadata."""
        weight_bits = cls.get_from_keys(config, ["w_bit", "bits"])
        group_size = cls.get_from_keys(config, ["q_group_size", "group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        return cls(weight_bits, group_size, zero_point)

    def get_linear_method(self) -> "AWQLinearMethod":
        """Create the AWQ linear method for this config."""
        return AWQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        """Return activation names that may appear in AWQ metadata."""
        return ["gelu", "gelu_fast", "gelu_new", "gelu_pytorch_tanh"]


class AWQLinearMethod(LinearMethodBase):
    """Linear-method implementation for AWQ-quantized weights."""
    def __init__(self, quant_config: AWQConfig):
        """Store AWQ packing metadata for later weight allocation."""
        self.quant_config = quant_config

    def create_weights(self, input_size: int, output_size: int) -> Dict[str, Any]:
        """Allocate AWQ-packed weight, zero-point, and scale tensors."""
        if input_size % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size."
            )
        if output_size % self.quant_config.pack_factor != 0:
            raise ValueError(
                "The output size is not aligned with the quantized weight shape. "
                "This can be caused by too large tensor parallel size."
            )

        qweight = Parameter(
            torch.empty(
                input_size,
                output_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {"input_dim": 0, "output_dim": 1, "packed_dim": 1,
             "pack_factor": self.quant_config.pack_factor},
        )

        qzeros = Parameter(
            torch.empty(
                input_size // self.quant_config.group_size,
                output_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {"input_dim": 0, "output_dim": 1, "packed_dim": 1,
             "pack_factor": self.quant_config.pack_factor},
        )

        scales = Parameter(
            torch.empty(
                input_size // self.quant_config.group_size,
                output_size,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {"input_dim": 0, "output_dim": 1})

        return {"qweight": qweight, "qzeros": qzeros, "scales": scales}

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the loaded quantized weights to the input activations."""
        qweight = weights["qweight"]
        qzeros = weights["qzeros"]
        scales = weights["scales"]

        pack_factor = self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (qweight.shape[-1] * pack_factor,)
        reshaped_x = x.reshape(-1, x.shape[-1])

        from .awq_triton import awq_gemm_triton

        out = awq_gemm_triton(reshaped_x, qweight, scales, qzeros, pack_factor)

        if bias is not None:
            out = out + bias
        return out.reshape(out_shape)
