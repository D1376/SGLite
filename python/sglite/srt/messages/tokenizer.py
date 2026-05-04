"""Messages exchanged with tokenizer workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sglite.sampling_params import SamplingParams

from .serialization import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    """Base class for messages exchanged with tokenizer workers."""

    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        """Serialize this tokenizer message for transport."""
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        """Deserialize a tokenizer message from transport data."""
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    """Batch of tokenizer messages delivered together."""
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    """Single generated token sent to the detokenizer."""
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    """Frontend text or chat request sent to tokenizer workers."""
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    """Abort request sent through tokenizer workers."""
    uid: int
