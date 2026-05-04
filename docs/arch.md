# Structure of SGLite

This document describes how the current repository is organized at runtime and how the code under `python/sglite/srt` is split by responsibility.

## Runtime Topology

Launching `python -m sglite` creates a small process graph rather than a single monolithic server.

### Frontend process

The parent process hosts either:

- the FastAPI server in `sglite.srt.entrypoints.api`
- the interactive CLI in `sglite.srt.entrypoints.cli`

Both paths share the same `FrontendManager`, which is responsible for:

- assigning request UIDs
- sending tokenizer-facing messages
- receiving incremental replies
- streaming output back to HTTP or CLI clients

### Tokenizer side

Tokenizer workers run `sglite.srt.tokenizer.server.tokenize_worker`.

Topology depends on `num_tokenizer`:

- `0`: one shared tokenizer/detokenizer worker
- `N > 0`: `N` tokenizer workers plus one dedicated detokenizer worker

The same worker loop handles three message types:

- tokenization requests
- detokenization requests
- abort requests

### Scheduler side

Each tensor-parallel rank runs one `Scheduler`, and each scheduler owns one `sglite.srt.model_executor.Engine`.

Rank responsibilities are asymmetric:

- rank 0 receives tokenized requests from the tokenizer side
- rank 0 forwards raw backend messages to the other ranks over ZMQ pub/sub when `tp_size > 1`
- all ranks execute the same scheduling decisions locally
- only rank 0 sends detokenization results back out

### Communication layers

SGLite uses two communication mechanisms:

- ZeroMQ for process-to-process control flow inside one host
- `torch.distributed` process groups for tensor-parallel coordination

When PyNCCL is enabled, SGLite still creates a CPU-side `gloo` group for synchronization and uses PyNCCL for bulk tensor exchange.

## Online Request Flow

The online path is a pipeline across frontend, tokenizer, scheduler, model executor, and detokenizer components.

1. A client sends a request to `/generate` or `/v1/chat/completions`.
2. `FrontendManager` allocates a UID and sends a `TokenizeMsg`.
3. The tokenizer worker converts raw text or chat messages into token IDs and emits a `UserMsg`.
4. Scheduler rank 0 receives that backend message. In multi-rank runs, it republishes the same raw message to the other ranks.
5. `PrefillManager` decides whether the request can enter the next prefill batch immediately or must be chunked.
6. `CacheManager` matches reusable prefixes, allocates cache pages, and updates the page table for the request slots selected by `TableManager`.
7. `sglite.srt.model_executor.Engine` prepares backend metadata, runs the model on the local GPU, and samples the next token.
8. Rank 0 wraps sampled tokens as `DetokenizeMsg` objects and sends them to the tokenizer side.
9. The detokenizer worker assembles incremental printable text and emits `UserReply`.
10. `FrontendManager` streams the reply to the HTTP client or CLI session.

Abort handling follows the same pipeline in reverse: the frontend sends `AbortMsg`, the tokenizer converts it into `AbortBackendMsg`, and the scheduler frees the request state if it is still live.

## Scheduling and Memory Model

The core scheduling logic is built from a few small state containers rather than one large scheduler object.

### Request state

`sglite.srt.request_state` defines the runtime request containers:

- `Req`: one in-flight generation request
- `Batch`: one scheduler-selected execution batch

Each `Req` tracks:

- host-side token IDs
- cached prefix length
- current device-visible length
- output budget
- request UID
- sampling parameters
- prefix-cache handle

`Batch` is a thin container populated by the scheduler immediately before execution with flattened token IDs, positions, output locations, and attention metadata.

### Request-slot tables

`sglite.srt.scheduler.table.TableManager` owns:

- the free list for request slots
- the shared token pool
- the request-to-page-table row mapping

This is the scheduler-side index that lets the runtime map one logical request to one reusable row in the global page table.

### Prefix cache and KV allocation

`sglite.srt.scheduler.cache.CacheManager` coordinates three things:

- prefix matching
- page allocation and eviction
- cache insertion after prefill

It sits above the lower-level `sglite.srt.mem_cache` abstractions:

- `BaseKVCachePool`
- `BasePrefixCache`
- `MHAKVCache`
- `RadixPrefixCache`
- `NaivePrefixCache`

The prefix cache works in token units, while allocation is page-aligned when `page_size > 1`.

### Prefill and decode queues

Scheduling is split by phase:

- `PrefillManager` manages queued requests that still need prompt ingestion
- `DecodeManager` manages active requests that are ready for the next decode token

Long prompts are chunked by `PrefillManager` through `ChunkedReq`, which lets one request span multiple scheduler turns before it becomes a normal decode request.

### Overlap scheduling

The scheduler uses two CUDA streams:

- one stream for metadata preparation
- one stream owned by the model executor for model execution

When overlap scheduling is enabled, the runtime prepares the next batch while the previous batch is still executing. This behavior is switched off only when `SGLITE_DISABLE_OVERLAP_SCHEDULING` is set.

## Model Executor Layout

`sglite.srt.model_executor.Engine` owns the rank-local execution stack.

### Initialization responsibilities

At startup, the model executor:

- sets tensor-parallel rank information
- resolves backend defaults such as `attention_backend` and `moe_backend`
- initializes distributed communication
- constructs the model on `meta` and then loads real weights
- allocates the KV cache pool
- allocates the page table
- builds the attention backend
- optionally builds the MoE backend
- creates the sampler
- prepares CUDA graph replay support

### Forward path

For each batch, the model executor:

1. enters the batch-scoped runtime `Context`
2. runs either the normal model forward or a CUDA-graph replay
3. samples next tokens on GPU
4. copies sampled tokens back to CPU asynchronously
5. returns both GPU and CPU views plus a CUDA event that marks copy completion

The scheduler consumes that result, updates request state, writes sampled tokens back into the token pool for future decode steps, and emits detokenization work.

## Package Layout

The Python implementation lives under `python/sglite/srt`.

### Public entry modules

- `sglite.__main__`: module entry point for `python -m sglite`
- `sglite.cli`: thin helper for launching directly into CLI mode
- `sglite.srt.request_state`: `Req` and `Batch`
- `sglite.srt.forward_context`: batch-scoped `Context` and global context helpers
- `sglite.sampling_params`: `SamplingParams`
- `sglite.srt.envs`: environment-backed runtime settings

### Frontend and protocol

- `sglite.srt.entrypoints.args`: CLI parsing and resolved `ServerArgs`
- `sglite.srt.entrypoints.launch`: process startup
- `sglite.srt.entrypoints.api`: FastAPI routes and bootstrap
- `sglite.srt.entrypoints.frontend_manager`: frontend-side request/reply coordination
- `sglite.srt.entrypoints.protocol`: request models and sampling-param conversion
- `sglite.srt.entrypoints.cli`: interactive terminal UI

### Scheduling and execution

- `sglite.srt.scheduler`: online/offline scheduler, I/O mixin, prefill/decode logic, cache manager, and slot tables
- `sglite.srt.model_executor`: model execution, CUDA graph handling, and token sampling
- `sglite.srt.forward_context`: batch-scoped `Context` and global context helpers
- `sglite.srt.request_state`: `Req` and `Batch`
- `sglite.sampling_params`: `SamplingParams`

### Model stack

- `sglite.srt.model_executor.models`: model registry, config normalization, AWQ quantization selection, and model-family implementations
- `sglite.srt.model_executor.model_loader`: checkpoint weight loading, sharding, projection merging, and expert packing
- `sglite.srt.model_executor.layers`: common transformer layers, tensor-parallel layers, RoPE, normalization, and quantized linear paths including AWQ and AWQ Marlin
- `sglite.srt.model_executor.layers.attention`: backend registry plus FlashAttention, FlashInfer, and TensorRT-LLM implementations
- `sglite.srt.model_executor.layers.fused_moe`: MoE backend registry and fused backend
- `sglite.srt.mem_cache`: KV-cache pools and prefix-cache implementations

### Distributed and Messaging Layers

- `sglite.srt.distributed`: tensor-parallel topology and communication helpers
- `sglite.srt.messages`: serialized message types for tokenizer, backend, and frontend traffic
- `sglite.srt.tokenizer`: tokenization, detokenization, and tokenizer worker entry points

### Low-level kernels and utilities

- `sglite.kernels`: custom CUDA/C++ kernels, JIT helpers, and bindings
- `sglite.srt.utils`: logging, tokenizer/model loading, ZMQ wrappers, registry helpers, and misc utilities
- `sglite.benchmark`: reusable benchmark client and kernel timing helpers

### Offline API

- `sglite.llm`: `LLM`, an in-process generation wrapper built on top of `Scheduler`

In offline mode, `SchedulerIOMixin` swaps out ZMQ-based I/O for local in-memory request intake and result collection.

## Repository Layout Outside `python/sglite/srt`

The rest of the repository is small and conventional:

- `tests/`: unit tests for core request state, env parsing, serialization, kernels, and scheduler behavior
- `benchmark/`: benchmarking scripts outside the Python package
- `assets/`: static assets such as the project logo
