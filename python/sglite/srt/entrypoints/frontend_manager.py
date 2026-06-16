"""Frontend coordination between clients and backend workers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

from fastapi import Request
from sglite.srt.messages import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from sglite.sampling_params import SamplingParams
from sglite.srt.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger

if TYPE_CHECKING:
    from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    """Normalize single and batched frontend replies into a flat list."""
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


@dataclass
class FrontendManager:
    """Bridges HTTP or CLI clients with tokenizer and backend workers."""
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        """Allocate state for a newly connected frontend user."""
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self) -> None:
        """Listen for incoming frontend requests and forward them downstream."""
        while True:
            msg = await self.recv_tokenizer.get()
            for reply in _unwrap_msg(msg):
                if reply.uid not in self.ack_map:
                    continue
                self.ack_map[reply.uid].append(reply)
                self.event_map[reply.uid].set()

    def _create_listener_once(self) -> None:
        """Start the background reply listener on first use."""
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg) -> None:
        """Send one frontend message to the tokenizer side."""
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def send_one_and_create_uid(
        self,
        *,
        text: str | List[Dict[str, str]],
        sampling_params: SamplingParams,
    ) -> int:
        """Allocate a user id, submit the request, and return that id."""
        uid = self.new_user()
        await self.send_one(
            TokenizeMsg(
                uid=uid,
                text=text,
                sampling_params=sampling_params,
            )
        )
        return uid

    async def wait_for_ack(self, uid: int) -> AsyncIterator[UserReply]:
        """Yield replies for one user until the request finishes."""
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            last_ack: UserReply | None = None
            for last_ack in pending:
                yield last_ack
            if last_ack and last_ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int) -> AsyncIterator[bytes]:
        """Encode `/generate` chunks as server-sent events."""
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int) -> AsyncIterator[bytes]:
        """Encode OpenAI-compatible chat-completion chunks."""
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(
        self,
        generator: AsyncIterator[bytes],
        request: Request,
        uid: int,
    ) -> AsyncIterator[bytes]:
        """Forward chunks and abort backend work if the client disconnects."""
        try:
            async for chunk in generator:
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int) -> None:
        """Drop local state and send an abort message for one user request."""
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self) -> None:
        """Release owned resources and stop background work."""
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()
