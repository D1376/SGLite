"""Tokenizer-side request management."""

from __future__ import annotations

from typing import List

import torch
from sglite.srt.messages import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    """Owns tokenizer calls for text and chat requests."""
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Store the tokenizer used by this worker."""
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        """Tokenize frontend text or chat messages into CPU int32 tensors."""
        prompts: List[str] = []
        for msg in msgs:
            if isinstance(msg.text, list):
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text
            prompts.append(prompt)

        if not prompts:
            return []

        encoded = self.tokenizer(prompts, add_special_tokens=True, padding=False)
        return [
            torch.tensor(input_ids, dtype=torch.int32)
            for input_ids in encoded["input_ids"]
        ]
