# bench/

Benchmarking scripts for SSD, SGLang, and vLLM.

## Spec Bench base model conversation history

Released on `MLSys-ver1.10`; see [실행과 인수인계](../../MLSYS_VER110_HANDOVER.md).

`duet_measure.py --input-protocol base-history` uses the
`specbench_base_token_history_v1` protocol for LayerSkip-Llama2 base checkpoints.
This is a documented base-model continuation experiment on Spec-Bench data,
not the official MT-Bench chat-quality evaluation or its Vicuna prompt wrapper.
The model card identifies the checkpoint as continued pretraining of Llama-2
base; Meta distinguishes raw completion prompts from Llama-2-Chat's `[INST]`
format. No Vicuna system prompt, role labels, or `[INST]` tags are added.

- Keep all 480 questions / 560 turns in the supplied order. A question's turns
  must execute sequentially on the same method/seed. Shard by complete question.
- Turn 0 preserves the dataset's frozen `input_ids`, checking the target's
  vocabulary and exact decoded raw text. For a source without frozen IDs, use
  the target tokenizer with `add_special_tokens=True`. These two input policies
  are recorded in the manifest; do not silently mix them in a comparison.
- Turn 1 input is exactly `turn0_input_ids + turn0_answer_ids_without_terminal_EOS
  + tokenizer.encode("\n\n" + question1, add_special_tokens=False)`. Thus the
  actual generated answer is retained without text decoding/re-encoding, and
  no extra BOS or EOS is injected at the boundary. The two-newline boundary is
  this experiment's convention; the base checkpoint has no official chat template.
- MT follow-ups ignore the old dataset's standalone follow-up `input_ids`.
  The other five tasks retain their original single-turn task prompts and IDs.
- Keep context 2048 and per-turn output cap `min(1024, 2048 - actual_input_length)`.
  Do not truncate questions, retrieved documents, or earlier answers. If no
  output capacity remains, log `context_limit_input` and do not call the model.
  Keep the engine's speculative-lookahead context protection; log its separate
  `context_limit_guard` reason. EOS, output cap and context cap are also separate.
- First three **dataset turns** are warmup and excluded from metrics. Their
  answers still enter the conversation: `82_t0` is warmup, and measured `82_t1`
  must use its answer. Warmup is not three extra requests.
- Preserve B1, target/draft temperature 0.7, unrestricted top-p/top-k, natural
  EOS, and the chosen dtype/DUET parameters. The wrapper retains the previous
  RNG reseeding at warm/measured phase boundaries without resetting history.
  Cold-prefix allocation remains enabled by default: each turn prefills its
  complete input; input history does not imply KV-cache reuse between requests.
- Store actual inputs and outputs, their hashes, prior-answer identity, effective
  cap, phase and termination reason in `turn_ledger`. Prior answers are input
  context in turn 1 and are not counted again as newly generated output tokens.
  Preparation/history assembly is outside the existing generation timer.
- Retain separate prefill and decode timings. Main reports use turn ratios,
  then equal question weights. Raw and existing filtered decode summaries are
  both retained. Unadmitted turns have no TPS and are explicitly listed; all
  cross-method comparisons must report valid populations, not silently zero-fill.

Export an input-format fixture **without running a model**:

```bash
python ssd/bench/specbench_history.py \
  --target /path/to/layerskip-llama2-7b \
  --prompts results/mlsys_ver02/equal_budget_20261014/SPEC_BENCH_FULL.json \
  --export /new/path/PROTOCOL_FIXTURE.json
```

Compare the dataset hash, frozen-ID policy, tokenizer file/backend hashes,
package versions, and fixture's exact input IDs across servers. The fixture
uses a synthetic fixed answer solely to check serialization; real methods use
their own generated answer, so real second-turn inputs need not be identical.
References must use the same serialization/input-ID policy before comparison.
`--input-protocol independent-raw` explicitly reproduces the old no-history
workload. Its results must not be labelled conversation-history MT-Bench.
`--task mt_bench` selects all 80 conversations / 160 turns for partial
remeasurement; the default `--task all` runs all 480 questions / 560 turns.
With history, `--limit N` selects the first N complete questions, not N spaced
turns. CPU regressions are in `ssd/tests/test_specbench_history.py`; no new GPU
performance result accompanies this protocol change.

After generation, validate actual inputs, answer links and accounting with
`python ssd/tools/verify_dispatch/check_history.py /path/to/run.json`.

Sources: [LayerSkip model card](https://huggingface.co/facebook/layerskip-llama2-7B),
[Meta base and Chat formats](https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/meta-llama-2).



We benchmark SGLang/vLLM via their server APIs rather than offline/in-process for two reasons:
1. Server mode exposes `/metrics` endpoints that offline mode doesn't.
2. We found server mode more performant in practice.

There is a small confound from HTTP overhead and prefill time being included in the
wall-clock measurement, but since this is a decode-heavy workload (512 output tokens,
each forward pass takes much longer than an HTTP round-trip), we find the difference to empirically be negligible.

## Setup

Use separate envs for all three stacks (`ssd`, `sglang`, `vllm`) because required FlashInfer versions by each conflict. 

```bash
# SSD (from repo root)
# uses pyproject.toml + uv.lock
uv sync

# SGLang 0.5.9
conda create -n sglang python=3.11
conda activate sglang
pip install "sglang[all]==0.5.9"

# vLLM 0.16.0
conda create -n vllm016 python=3.11
conda activate vllm016
pip install "vllm==0.16.0"
```

SGLang/vLLM envs also need: `transformers`, `aiohttp`, `wandb` (optional).

## Model paths

SGLang/vLLM baselines read from `bench_paths.py`. Edit it or set env vars:
```bash
export SSD_HF_CACHE=/path/to/huggingface/hub
export BENCH_LLAMA_70B=/path/to/Llama-3.3-70B-Instruct
export BENCH_LLAMA_1B=/path/to/Llama-3.2-1B-Instruct
# etc.
```

SSD reads from `ssd/paths.py` (separate, env vars documented there).

## Datasets

4 datasets (humaneval, alpaca, gsm8k, ultrafeedback), 128 prompts each with `--all`.
Set `SSD_DATASET_DIR` or edit `ssd/paths.py`. Generate with `scripts/get_data_from_hf.py`.

## Usage

### SSD (native, no server)
`bench.py` and `bench_helpers.py` are SSD-only (import from `ssd.*`).
```bash
python -O bench.py --llama --size 70 --async --spec --k 7 --f 3 --b 1 \
    --temp 0 --numseqs 128 --output_len 512 --all --gpus 5
```

### SGLang
```bash
conda activate sglang
# Speculative decoding (default):
python bench/run_sglang_bench.py --llama
# Autoregressive baseline:
python bench/run_sglang_bench.py --llama --mode ar
# With wandb logging:
python bench/run_sglang_bench.py --llama --wandb --group mygroup --name myrun
```

### vLLM
```bash
conda activate vllm016
# Speculative decoding (default):
python bench/run_vllm_bench.py --llama
# Autoregressive baseline:
python bench/run_vllm_bench.py --llama --mode ar
```

## Troubleshooting

If you see Triton errors like `OSError: [Errno 116] Stale file handle`, set cache dirs to local disk (not network filesystems):

```bash
export TRITON_CACHE_DIR=/scratch/$USER/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/$USER/torchinductor_cache
```

If a run crashes or you Ctrl-C, stale worker processes can linger and hold resources.
Before rerunning, kill them and verify GPUs are free:

```bash
pkill -9 -f "bench.py|vllm.entrypoints|sglang.launch_server|VLLM::|sglang::"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```
