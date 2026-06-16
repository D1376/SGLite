"""FastAPI endpoints and server bootstrap logic."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sglite.srt.messages import BaseFrontendMsg, BaseTokenizerMsg
from sglite.srt.utils import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    configure_external_loggers,
    init_logger,
)

from .args import ServerArgs
from .frontend_manager import FrontendManager
from .protocol import (
    GenerateRequest,
    ModelCard,
    ModelList,
    OpenAICompletionRequest,
    resolve_completion_prompt,
)
from .cli import run_cli as run_cli_session

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE: FrontendManager | None = None


def get_global_state() -> FrontendManager:
    """Return the singleton frontend manager after startup has initialized it."""
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


async def _submit_request(
    state: FrontendManager,
    *,
    text: str | list[dict[str, str]],
    max_tokens: int,
    ignore_eos: bool,
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 1.0,
) -> int:
    """Submit one text or chat request to the tokenizer and return its user id."""
    from .protocol import make_sampling_params

    return await state.send_one_and_create_uid(
        text=text,
        sampling_params=make_sampling_params(
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ),
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Manage FastAPI startup and shutdown hooks for the frontend manager."""
    yield
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="SGLite API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    """Handle the simple streaming `/generate` endpoint."""
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = await _submit_request(
        state,
        text=req.prompt,
        ignore_eos=req.ignore_eos,
        max_tokens=req.max_tokens,
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    """Return a simple health payload for the compatibility root."""
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    """Handle OpenAI-compatible chat-completion requests."""
    state = get_global_state()
    uid = await _submit_request(
        state,
        text=resolve_completion_prompt(req),
        ignore_eos=req.ignore_eos,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
        media_type="text/event-stream",
    )


@app.get("/v1/models")
async def available_models():
    """Return the single configured model in OpenAI-compatible format."""
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_cli: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_cli: If True, run an interactive terminal CLI instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_cli:
        assert not config.use_dummy_weight, "CLI mode does not support dummy weights."

    configure_external_loggers()

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    start_backend()

    logger.info(
        "Frontend ready: mode=%s host=%s port=%d model=%s tp_size=%d",
        "cli" if run_cli else "server",
        host,
        port,
        config.model_path,
        config.tp_info.size,
    )
    if not run_cli:
        uvicorn.run(app, host=host, port=port, log_config=None)
    else:
        asyncio.run(run_cli_session(_GLOBAL_STATE))
