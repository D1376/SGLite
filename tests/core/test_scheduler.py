"""Tests for scheduler."""

from __future__ import annotations

import torch
import multiprocessing as mp
from transformers import AutoTokenizer

from sglite.srt.distributed import DistributedInfo
from sglite.srt.messages import BaseBackendMsg, BaseTokenizerMsg, DetokenizeMsg, ExitMsg, UserMsg
from sglite.sampling_params import SamplingParams
from sglite.srt.scheduler import Scheduler, SchedulerConfig
from sglite.srt.utils import ZmqPullQueue, ZmqPushQueue, init_logger

logger = init_logger(__name__)


@torch.inference_mode()
def scheduler(config: SchedulerConfig, queue: mp.Queue) -> None:
    scheduler = Scheduler(config)
    queue.put(None)
    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger.info_rank0("Scheduler exiting...")


def main():
    config = SchedulerConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_reqs=4,
        cuda_graph_bs=[2, 4, 8],
    )

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    p = mp.Process(target=scheduler, args=(config, q))
    p.start()
    q.get()

    send_backend = ZmqPushQueue(
        config.zmq_backend_addr,
        create=False,
        encoder=BaseBackendMsg.encoder,
    )

    recv_backend = ZmqPullQueue(
        config.zmq_detokenizer_addr,
        create=False,
        decoder=BaseTokenizerMsg.decoder,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    prompt = "What's the answer to life, the universe, and everything?"
    ids = tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
    send_backend.put(
        UserMsg(
            uid=0,
            input_ids=ids,
            sampling_params=SamplingParams(max_tokens=100),
        )
    )

    while True:
        msg = recv_backend.get()
        assert isinstance(msg, DetokenizeMsg)
        ids = torch.cat([ids, torch.tensor([msg.next_token], dtype=torch.int32)])
        if msg.finished:
            break

    logger.info(tokenizer.decode(ids.tolist()))
    send_backend.put(ExitMsg())


if __name__ == "__main__":
    main()
