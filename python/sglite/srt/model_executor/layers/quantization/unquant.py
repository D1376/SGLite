"""Unquantized linear method implementation."""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .base_config import LinearMethodBase, set_weight_attrs


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear-method implementation for regular unquantized weights."""
    def __init__(self, separate_bias_add: bool = False):
        """Store whether bias should be added outside `F.linear`."""
        self.separate_bias_add = separate_bias_add

    def create_weights(self, input_size: int, output_size: int) -> Dict[str, Any]:
        """Allocate a dense floating-point weight matrix."""
        weight = Parameter(
            torch.empty(output_size, input_size),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        return {"weight": weight}

    def apply_weights(
        self,
        weights: Dict[str, Any],
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the loaded dense weights to the input activations."""
        weight = weights["weight"]
        if self.separate_bias_add:
            if bias is not None:
                return F.linear(x, weight) + bias
            return F.linear(x, weight)
        return F.linear(x, weight, bias)
