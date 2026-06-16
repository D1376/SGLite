# SGLite Features

This document describes the features that are implemented in the current codebase. It is intentionally grounded in the behavior under `python/sglite/srt`, not in roadmap items or upstream SGLang capabilities that have not been wired into this repository.

## Serving Modes

### OpenAI-style server

The main entry point is:

```bash
python -m sglite --model <model-path-or-repo>
```

This launches a FastAPI frontend plus backend worker processes. The frontend exposes:

- `POST /generate`
- `GET|POST|HEAD|OPTIONS /v1`
- `POST /v1/chat/completions`
- `GET /v1/models`

`/v1/models` reports the single model configured for the running server. Both generation endpoints return streamed responses over `text/event-stream`.

### Interactive CLI

SGLite can run the same runtime behind an interactive terminal CLI:

```bash
python -m sglite --model <model-path-or-repo> --cli
```

The CLI keeps multi-turn history in memory, converts that history into chat messages, and supports:

- `/clear` to reset chat history
- `/quit` to exit

CLI mode defaults a few runtime settings for predictable local interaction:

- `max_running_req=1` when `--max-running-requests` is not explicitly set
- `cuda_graph_max_bs=1` when `--cuda-graph-max-bs` / `--graph` is not explicitly set
- `silent_output=True`

`--dummy-weight` is explicitly rejected in CLI mode.

### Offline Python API

SGLite also exposes an in-process interface through `sglite.llm.LLM`. This path reuses the scheduler and model-executor stack without starting the HTTP frontend or tokenizer subprocesses.

The offline API accepts either raw strings or token IDs and returns both generated text and token IDs.

### Tensor parallel serving

Multi-GPU inference is supported through tensor parallelism:

- `--tp-size`
- `--tensor-parallel-size`

SGLite launches one scheduler/model-executor process per TP rank. Rank 0 handles external request I/O, and all ranks participate in model execution.

## Model Loading

### Model sources

`--model` accepts either:

- a local model directory
- a Hugging Face repo ID

`--model-source` controls remote resolution:

- `huggingface`
- `modelscope`

When `--model-source modelscope` is used with a repo ID, SGLite downloads the snapshot before startup.

### DType resolution

Supported dtypes are:

- `auto`
- `float16`
- `bfloat16`
- `float32`

`auto` resolves from the Hugging Face config loaded for the model.

### Registered model families

The current registry supports these architecture names:

- `LlamaForCausalLM`
- `Qwen2ForCausalLM`
- `Qwen3ForCausalLM` / `Qwen3MoeForCausalLM`
- `MistralForCausalLM`
- `Mistral3ForConditionalGeneration`

In practice, that covers the Llama, Qwen2/Qwen2.5, Qwen3 / Qwen3 MoE, and Mistral families when their configs expose one of the registered architecture names.

AWQ-quantized variants of those supported model families are also supported. SGLite detects AWQ checkpoints from Hugging Face config metadata or local quantization config files, prefers the Marlin path for compatible checkpoints, and falls back to the Triton AWQ implementation when Marlin requirements are not satisfied. AWQ Marlin requires `SM80+` GPUs.

### Dummy weights for smoke tests

`--dummy-weight` skips real weight loading and fills the model state dict with random tensors. This is useful for startup-path testing, not for meaningful inference quality.

### AWQ quantization detection

SGLite detects AWQ quantization from either:

- `quantization_config` on the Hugging Face config
- `quant_config.json`
- `quantize_config.json`

When the configuration is compatible, SGLite prefers the Marlin path; otherwise it falls back to the Triton AWQ implementation.

## Request Interfaces

### `/generate`

`/generate` is the smallest request surface. The request model contains:

- `prompt`
- `max_tokens`
- `ignore_eos`

The response is a simple SSE stream of incremental text chunks followed by `[DONE]`.

### `/v1/chat/completions`

The OpenAI-style route requires `model` for OpenAI-compatible request shape, but the value is not used for routing. It accepts either:

- `messages`
- `prompt`

When `messages` are provided, SGLite converts them with the tokenizer chat template using `add_generation_prompt=True`.

The current implementation actively uses these generation controls:

- `max_tokens`
- `temperature`
- `top_k`
- `top_p`
- `ignore_eos`

The request model also accepts several OpenAI-compatible fields that are not wired into generation behavior yet:

- `n`
- `stream`
- `stop`
- `presence_penalty`
- `frequency_penalty`

The route still responds as a streaming SSE endpoint even if `stream=false` is sent.

### `/v1`

`/v1` is a lightweight compatibility root that returns a health payload:

```json
{"status": "ok"}
```

## Runtime Behavior

### Chunked prefill

Long prompt ingestion is capped by `--max-prefill-length` (alias `--max-extend-length`). Requests that exceed the current prefill budget are split across iterations so the scheduler can admit them incrementally.

### KV cache and prefix reuse

The runtime allocates a paged KV cache and manages it through:

- `MHAKVCache` for the device-side KV pool
- `radix` prefix cache
- `naive` prefix cache

`radix` is the default cache strategy. Prefix matches are used during prefill so previously seen prompt prefixes can be reused instead of recomputed.

The main cache-related knobs are:

- `--cache-type`
- `--page-size`
- `--num-pages`
- `--mem-frac`

### Attention backends

SGLite currently exposes three attention backend names:

- `fa` for FlashAttention
- `fi` for FlashInfer
- `trtllm` for TensorRT-LLM FMHA

`--attn-backend` accepts either:

- a single backend name
- a `prefill,decode` pair such as `fa,fi`
- `auto`

`auto` resolves to the backend combination selected by `sglite.srt.model_executor.engine._adjust_config` for the current GPU generation.

Backend-specific constraints that are enforced in the runtime:

- `trtllm` only supports page sizes `16`, `32`, and `64`
- when `trtllm` is selected with an unsupported page size, SGLite normalizes it to `64`

### CUDA graph capture

Decode execution can be captured and replayed through CUDA graphs. The main control is:

- `--cuda-graph-max-bs`

Graph capture is disabled by setting the max batch size to `0`.

### MoE execution

For MoE models, `--moe-backend` selects the execution backend. The built-in backend registry currently contains:

- `fused`

When `--moe-backend auto` is used on an MoE model, SGLite resolves it to `fused`.

### Overlap scheduling

The scheduler overlaps metadata preparation on one CUDA stream with model execution on another stream. This is enabled by default.

It can be disabled with:

- `SGLITE_DISABLE_OVERLAP_SCHEDULING=1`

### Tokenizer topology

Tokenizer workers are controlled with:

- `--num-tokenizer`
- `--tokenizer-count`

Behavior depends on the value:

- `0`: one shared tokenizer/detokenizer worker
- `N > 0`: `N` tokenizer workers plus one dedicated detokenizer worker

### Distributed communication

For TP runs, SGLite always creates a CPU-side `gloo` process group for coordination. Tensor exchange uses either:

- PyNCCL helpers by default
- `torch.distributed` NCCL collectives when `--disable-pynccl` is passed

## Environment Variables

The current environment-backed settings are:

- `SGLITE_CLI_MAX_TOKENS`
- `SGLITE_CLI_TOP_K`
- `SGLITE_CLI_TOP_P`
- `SGLITE_CLI_TEMPERATURE`
- `SGLITE_FLASHINFER_USE_TENSOR_CORES`
- `SGLITE_DISABLE_OVERLAP_SCHEDULING`
- `SGLITE_PYNCCL_MAX_BUFFER_SIZE`
