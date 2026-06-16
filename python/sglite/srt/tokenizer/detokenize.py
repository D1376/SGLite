"""Detokenization helpers and output assembly."""

from dataclasses import dataclass
from typing import Dict, List

from sglite.srt.messages import DetokenizeMsg
from transformers import PreTrainedTokenizerBase

# Adapted from SGLang streaming detokenization.


def _is_chinese_char(cp: int):
    """Return whether a codepoint belongs to a CJK ideograph block."""
    # Treat codepoints in the CJK Unified Ideographs blocks as Chinese characters:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B820 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
    ):
        return True

    return False


def find_printable_text(text: str):
    """Return the longest printable prefix that avoids partial words."""
    # Adapted from Hugging Face's TextIteratorStreamer printable-text heuristic.

    # Flush complete lines immediately.
    if text.endswith("\n"):
        return text
    # CJK ideographs do not need whitespace to form complete output units.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Keep one trailing character when only the previous character is definitely complete.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # For space-delimited text, emit only through the last whitespace boundary.
    else:
        return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    """Tracks streaming decode state for one request."""
    decoded_ids: List[int]
    decoded_str: str
    read_offset: int  # Number of decoded ids consumed by batch_decode.
    surr_offset: int  # Start offset for the surrogate decode window.
    sent_offset: int  # Number of characters already emitted to the client.


class DetokenizeManager:
    """Converts generated token ids into incremental frontend text chunks."""
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Create per-request decode state for the supplied tokenizer."""
        # Per-request streaming decode state.
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """Decode generated tokens and return incremental text per request."""
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                s.decoded_ids.append(msg.next_token)
            read_ids.append(s.decoded_ids[s.surr_offset :])
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str) :]
            # Accept the new suffix only when it does not end with a replacement character.
            if len(new_text) > 0 and not new_text.endswith("�"):
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            incremental_output = output_str[s.sent_offset :]
            s.sent_offset = len(output_str)
            incremental_strs.append(incremental_output)
            if msg.finished:
                del self.decode_map[msg.uid]

        return incremental_strs
