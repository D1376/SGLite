"""Base module helpers for the layer stack."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    """Join a state-dict prefix and field name."""
    return f"{prefix}.{name}" if prefix else name


class BaseModule:
    """Base class for lightweight modules in the model stack."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the module and return its output tensors."""
        ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """Return the state-dict entries owned by this module."""
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseModule):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """Load tensors from a flat state dict into this module tree."""
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)
            elif isinstance(param, BaseModule):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StatelessModule(BaseModule):
    """Module base class for layers without trainable parameters."""

    def __init__(self):
        """Initialize the stateless base class."""
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """Reject unexpected weights for stateless modules."""
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """Return the state-dict entries owned by this module."""
        return result if result is not None else {}


T = TypeVar("T", bound=BaseModule)


class ModuleList(BaseModule, Generic[T]):
    """Small list-like container for child modules."""
    def __init__(self, modules: List[T]):
        """Store child modules without adding torch.nn.Module machinery."""
        super().__init__()
        self.modules = modules

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """Return the state-dict entries owned by this module."""
        result = result if result is not None else {}
        for i, module in enumerate(self.modules):
            module.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """Load weights into every child module by numeric prefix."""
        for i, module in enumerate(self.modules):
            module.load_state_dict(
                state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True
            )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
