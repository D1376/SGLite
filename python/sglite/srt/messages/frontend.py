"""Messages sent from the scheduler back to frontend clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .serialization import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    """Base class for messages emitted back to frontend clients."""

    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        """Serialize this frontend reply for transport."""
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        """Deserialize a frontend reply from transport data."""
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    """Batch of frontend replies delivered together."""
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    """Frontend-facing token reply emitted for one request."""
    uid: int
    incremental_output: str
    finished: bool
