"""Tests for tokenizer-side batching."""

from __future__ import annotations

from typing import Dict, List

from sglite.srt.messages import TokenizeMsg
from sglite.sampling_params import SamplingParams
from sglite.srt.tokenizer.tokenize import TokenizeManager


class DummyTokenizer:
    """Small tokenizer double that exposes batched calls."""

    def __init__(self) -> None:
        self.batch_calls = 0

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[-1]["content"] + "<assistant>"

    def __call__(
        self,
        prompts: List[str],
        *,
        add_special_tokens: bool,
        padding: bool,
    ) -> Dict[str, List[List[int]]]:
        assert add_special_tokens is True
        assert padding is False
        self.batch_calls += 1
        return {"input_ids": [[len(prompt), index] for index, prompt in enumerate(prompts)]}


def test_tokenize_batches_prompts_in_one_tokenizer_call():
    tokenizer = DummyTokenizer()
    manager = TokenizeManager(tokenizer)  # type: ignore[arg-type]
    params = SamplingParams(max_tokens=4)

    tensors = manager.tokenize(
        [
            TokenizeMsg(uid=1, text="hello", sampling_params=params),
            TokenizeMsg(
                uid=2,
                text=[{"role": "user", "content": "chat"}],
                sampling_params=params,
            ),
        ]
    )

    assert tokenizer.batch_calls == 1
    assert [tensor.tolist() for tensor in tensors] == [[5, 0], [15, 1]]
