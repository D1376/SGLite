"""Interactive CLI helpers for the frontend server."""

from __future__ import annotations

import asyncio
import time
from typing import List, Tuple

from fastapi.responses import StreamingResponse
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from sglite.srt.envs import ENV
from sglite.srt.utils import dim_text
from starlette.background import BackgroundTask

from .frontend_manager import FrontendManager
from .protocol import (
    Message,
    OpenAICompletionRequest,
    history_messages,
    make_sampling_params,
    message_dicts,
)


async def cli_completion(
    state: FrontendManager,
    req: OpenAICompletionRequest,
) -> StreamingResponse:
    """Render completion suggestions for the interactive CLI."""
    assert req.messages is not None, "CLI completion only supports chat-completions"
    uid = await state.send_one_and_create_uid(
        text=message_dicts(req.messages),
        sampling_params=make_sampling_params(
            ignore_eos=req.ignore_eos,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        ),
    )

    async def _abort():
        """Cancel the backend request when the response stream closes."""
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(_abort),
    )


async def run_cli(state: FrontendManager) -> None:
    """Run the interactive CLI loop for local inference."""
    commands = ["/quit", "/clear"]
    completer = WordCompleter(commands)
    session = PromptSession(">>> ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/quit":
                    return
                if cmd == "/clear":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")

            req = OpenAICompletionRequest(
                model="",
                messages=history_messages(history) + [Message(role="user", content=cmd)],
                max_tokens=ENV.CLI_MAX_TOKENS.value,
                top_k=ENV.CLI_TOP_K.value,
                top_p=ENV.CLI_TOP_P.value,
                temperature=ENV.CLI_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            token_count = 0
            t_start = time.perf_counter()
            async for chunk in (await cli_completion(state, req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                token_count += 1
                cur_msg += msg
                print(msg, end="", flush=True)
            elapsed = time.perf_counter() - t_start
            tps = token_count / elapsed if elapsed > 0 else 0.0
            print(dim_text(f"\n[{token_count} tokens | {tps:.1f} tok/s]"), flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        pass
    finally:
        print("Exiting CLI...")
        await asyncio.sleep(0.1)
        state.shutdown()

        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()
