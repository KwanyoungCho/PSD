# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SSD (Speculative Speculative Decoding)** is a research LLM inference engine implementing a novel speculative decoding algorithm where a draft model runs on a separate GPU in parallel with target model verification — enabling the draft to pre-compute outputs for multiple possible verification outcomes before results arrive. Published at ICLR 2026 (arXiv:2603.03251).

## Environment Setup

Requires Python 3.11+, CUDA >= 12.8, tested on H100s.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

Key environment variables:
- `SSD_HF_CACHE`: HuggingFace model cache directory
- `SSD_DATASET_DIR`: Processed benchmark datasets directory
- `SSD_CUDA_ARCH`: GPU architecture (9.0=H100, 8.0=A100, 8.9=L40/4090)

## Running the Code

**Interactive chat:**
```bash
cd bench
python -O chat.py --ssd --spec --async --k 7 --f 3 --gpus 5 --metrics
```

**Benchmark:**
```bash
cd bench
python -O bench.py --llama --size 70 --async --spec --k 7 --f 3 \
  --b 1 --temp 0 --numseqs 128 --output_len 512 --all --gpus 5
```

**Download models/datasets:**
```bash
python scripts/download_from_hf.py
python scripts/get_data_from_hf.py
```

Use `-O` flag with Python to enable optimizations (disables assertions).

## Architecture

### Core Inference Flow

```
LLM (ssd/llm.py)
  └── LLMEngine (ssd/engine/llm_engine.py)
        ├── ModelRunner(s)  — target model, rank 0 + TP workers (ssd/engine/model_runner.py)
        ├── DraftRunner     — draft model on separate GPU for async SSD (ssd/engine/draft_runner.py)
        ├── Scheduler       — request batching & block memory management (ssd/engine/scheduler.py)
        ├── Verifier        — validates draft tokens against target logits (ssd/engine/verifier.py)
        └── Speculator      — generates draft token candidates
              ├── SpeculatorSync  — sequential draft/verify on same GPU
              └── SpeculatorAsync — parallel draft on separate GPU with tree caching
```

### Three Inference Modes

1. **Autoregressive** (`AutoRegressiveStep`): Standard AR baseline
2. **Sync speculative decoding** (`SpecDecodeStep` + `SpeculatorSync`): Draft and target alternate on the same GPU
3. **Async SSD** (`SpecDecodeStep` + `SpeculatorAsync`): Core contribution — draft runs on a dedicated GPU and pre-computes a tree of candidate continuations, indexed by possible verification outcomes

### Key Modules

- **`ssd/config.py`**: `Config` dataclass controlling all inference parameters (model paths, KV cache settings, speculation k/fan-out, EAGLE3 support). Validation in `__post_init__`.
- **`ssd/paths.py`**: Default model paths and environment variable resolution. `DEFAULT_TARGET` = Llama-3.1-8B-Instruct, `DEFAULT_DRAFT` = Llama-3.2-1B-Instruct.
- **`ssd/engine/helpers/cudagraph_helpers.py`**: CUDAGraph capture/replay — major latency optimization. Critical to understand when modifying decode-phase execution.
- **`ssd/utils/async_helpers/`**: NCCL communication between draft/target processes, async spec tree caching, recovery token logic.
- **`ssd/utils/context.py`**: Thread-local context holding prefill/decode state (attention masks, block tables, `cu_seqlens`) — passed implicitly through layers.
- **`ssd/layers/attention.py`**: FlashInfer-backed paged attention with KV cache.

### Model Implementations

- `ssd/models/llama3.py`: LlamaForCausalLM with tensor parallelism
- `ssd/models/qwen3.py`: Qwen3ForCausalLM
- `ssd/models/eagle3_draft_llama3.py`: EAGLE3 lightweight draft model

### Memory Management

`Scheduler` manages paged KV cache via `BlockManager`. Blocks are allocated per-sequence for both target and draft models. The lookahead budget (how many tokens to speculate) is computed each step based on available memory.

## Speculative Decoding Step Lifecycle

1. `scheduler.schedule()` — select sequences, determine prefill vs decode mode
2. **Prefill phase**: target encodes full prompt; optionally draft warms its cache
3. **Decode loop**:
   - `Speculator.speculate()` — draft model generates k candidate tokens (sync) or a tree of candidates (async)
   - `Verifier.verify()` — target model evaluates all candidates in one forward pass
   - Accepted tokens are committed; a recovery token is sampled at the first rejection point
4. `log_metrics()` — throughput, token acceptance rate, cache hit rate

## Debugging & Profiling

- `verbose=True` in `Config`: per-step logging
- `debug_mode=True`: saves draft inputs during prefill
- `SSD_PROFILE=1`: timing profiler for bottleneck identification
- `max_steps` in `Config`: limit steps for quick testing

## Benchmarking

`bench/bench.py` benchmarks across HumanEval, Alpaca, C4, GSM8k, UltraFeedback datasets (128 seqs each). Comparison baselines require separate conda environments for SGLang 0.5.9 and vLLM 0.16.0 (`bench/run_sglang_bench.py`, `bench/run_vllm_bench.py`). WandB logging is optionally integrated.
