"""Messages sent from frontend components to the scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from sglite.sampling_params import SamplingParams

from .serialization import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    """Base class for messages consumed by scheduler and backend workers."""
    def encoder(self) -> Dict:
        """Serialize this backend message for transport."""
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        """Deserialize a backend message from transport data."""
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """Batch of backend messages delivered together."""
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    """Control message that asks backend workers to shut down."""
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    """Tokenized user request sent to the scheduler."""
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    """Abort request sent to the scheduler."""
    uid: int
