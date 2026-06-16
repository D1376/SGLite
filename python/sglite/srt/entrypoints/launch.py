"""Process launch helpers for the inference server."""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from sglite.srt.distributed import DistributedInfo
from sglite.srt.utils import init_logger, print_banner

if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    """Run one tensor-parallel scheduler worker in a subprocess."""
    import torch
    from sglite.srt.scheduler import Scheduler

    with torch.inference_mode():
        scheduler = Scheduler(args)
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def _start_scheduler_processes(server_args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    """Start all tensor-parallel scheduler subprocesses."""
    world_size = server_args.tp_info.size
    for rank in range(world_size):
        args = replace(
            server_args,
            tp_info=DistributedInfo(rank, world_size),
        )
        mp.Process(
            target=_run_scheduler,
            args=(args, ack_queue),
            daemon=False,
            name=f"sglite-TP{rank}-scheduler",
        ).start()


def _start_tokenize_process(
    server_args: ServerArgs,
    ack_queue: mp.Queue[str],
    *,
    addr: str,
    tokenizer_id: int,
    name: str,
) -> None:
    """Start one tokenizer or detokenizer worker subprocess."""
    from sglite.srt.tokenizer import tokenize_worker

    mp.Process(
        target=tokenize_worker,
        kwargs={
            "tokenizer_path": server_args.model_path,
            "addr": addr,
            "backend_addr": server_args.zmq_backend_addr,
            "frontend_addr": server_args.zmq_frontend_addr,
            "local_bs": 1,
            "create": server_args.tokenizer_create_addr,
            "tokenizer_id": tokenizer_id,
            "ack_queue": ack_queue,
        },
        daemon=False,
        name=name,
    ).start()


def _wait_until_ready(logger: logging.Logger, ack_queue: mp.Queue[str], server_args: ServerArgs) -> None:
    """Wait for worker acknowledgements and log a compact readiness summary."""
    expected_acks = server_args.num_tokenizer + 2
    for _ in range(expected_acks):
        ack = ack_queue.get()
        logger.debug("Startup ack: %s", ack)

    logger.info(
        "Backend workers ready: tp_size=%d tokenizers=%d detokenizer=1",
        server_args.tp_info.size,
        server_args.num_tokenizer,
    )


def launch_server(run_cli: bool = False) -> None:
    """Launch the server frontend and backend worker processes."""
    from .api import run_api_server
    from .args import parse_args

    server_args, run_cli = parse_args(sys.argv[1:], run_cli)
    logger = init_logger(__name__, "initializer")
    print_banner()

    def start_subprocess() -> None:
        """Start backend worker subprocesses and wait until they are ready."""
        mp.set_start_method("spawn", force=True)

        # Every worker posts one readiness ack before the frontend starts serving traffic.
        ack_queue: mp.Queue[str] = mp.Queue()

        _start_scheduler_processes(server_args, ack_queue)

        num_tokenizers = server_args.num_tokenizer
        # The detokenizer is a tokenizer worker running on the dedicated output channel.
        _start_tokenize_process(
            server_args,
            ack_queue,
            addr=server_args.zmq_detokenizer_addr,
            tokenizer_id=num_tokenizers,
            name="sglite-detokenizer-0",
        )
        for tokenizer_id in range(num_tokenizers):
            _start_tokenize_process(
                server_args,
                ack_queue,
                addr=server_args.zmq_tokenizer_addr,
                tokenizer_id=tokenizer_id,
                name=f"sglite-tokenizer-{tokenizer_id}",
            )

        # Only the primary scheduler rank posts readiness, plus every tokenizer and detokenizer.
        _wait_until_ready(logger, ack_queue, server_args)

    run_api_server(server_args, start_subprocess, run_cli=run_cli)


if __name__ == "__main__":
    launch_server()
