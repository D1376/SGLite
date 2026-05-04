"""Request and response models for the OpenAI-style API."""

from __future__ import annotations

import time
from typing import Dict, List, Literal, Tuple

from pydantic import BaseModel, Field
from sglite.sampling_params import SamplingParams


class GenerateRequest(BaseModel):
    """Request model for the lightweight `/generate` endpoint."""
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    """Single chat message entry used by the OpenAI-style protocol."""
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = Field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    """Model metadata returned by the compatibility API."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "SGLite"
    root: str


class ModelList(BaseModel):
    """Container for the available model cards."""
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


def message_dicts(messages: List[Message]) -> List[Dict[str, str]]:
    """Normalize chat messages into plain dictionaries."""
    return [msg.model_dump() for msg in messages]


def resolve_completion_prompt(req: OpenAICompletionRequest) -> str | List[Dict[str, str]]:
    """Resolve an OpenAI-style completion request into the internal prompt format."""
    if req.messages is not None:
        return message_dicts(req.messages)
    assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
    return req.prompt


def make_sampling_params(
    *,
    max_tokens: int,
    ignore_eos: bool,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
) -> SamplingParams:
    """Build SamplingParams from frontend request fields."""
    return SamplingParams(
        ignore_eos=ignore_eos,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def history_messages(history: List[Tuple[str, str]]) -> List[Message]:
    """Return the non-system messages in a chat history."""
    result: List[Message] = []
    for user_msg, assistant_msg in history:
        result.append(Message(role="user", content=user_msg))
        result.append(Message(role="assistant", content=assistant_msg))
    return result
