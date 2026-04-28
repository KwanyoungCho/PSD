# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a research monorepo with two distinct pieces:

- `ssd/` — a fork of the Speculative Speculative Decoding (SSD) engine
  (https://github.com/tanishqkumar/ssd). Python package `ssd/ssd/`,
  benchmark harness `ssd/bench/`. Extended with **MESA-SSD** (early-exit
  proxy speculative decoding) and an **AWQ W4A16** quantization backend
  built on sgl-kernel Marlin.
- Top-level `.py` scripts (`correction_analysis.py`, `ee_verify_analysis.py`,
  `plot_*.py`) — offline distribution-level analysis of target early-exit
  vs. draft/final distributions (JSD/KL/TVD, top-k overlap, recovery
  feasibility). These use HuggingFace `AutoModelForCausalLM` directly and
  are independent of the SSD engine. Outputs land in `results/`, `report/`,
  `report_llama70B/`, `layer_skip/`.

Three+ separate conda envs are involved:
- `PSD` — top-level analysis scripts (HF Transformers-based).
- `ssd` — SSD engine (built from `ssd/pyproject.toml` via `uv sync`;
  `bench/*.sh` call it at `/home/chokwans99/anaconda3/envs/ssd/bin/python`).
  torchao 0.12.0 must be added manually on top of pyproject pins.
- `sglang`, `vllm016` — baselines only, required by `bench/run_sglang_bench.py`
  and `bench/run_vllm_bench.py` (FlashInfer versions conflict with SSD, so
  they must live in separate envs).
- `awq-quant` — AutoAWQ calibration only (`scripts/awq_calibrate_autoawq.py`).
  AutoAWQ pins conflict with SSD's torch / transformers; mirror the
  sglang/vllm split.

## Active workstream

Current branch: **`feat/mesa-phase2-hybrid`**. The Phase 2 hybrid redesign
work is now consolidated under `ssd/docs/mesa/`:

- `ssd/docs/mesa/01-design.md` — MESA design (Parts 1-4: TreeLayout, Budget
  Split, Split CG, Rev1 Policy A/B; **Part 5: Phase 2 Hybrid v1** — terminology,
  5-region scratch, 8 CG buckets, `HybridPhase2Plan`, Step 0..9D progression).
- `ssd/docs/mesa/02-impl-issues.md` — implementation issue tracker (Parts 1-2:
  v1, Rev1; **Part 3: Phase 2 Hybrid Step 1..9D** + sync fix + per-depth label
  fix + 9D build hot-path optimization).
- `ssd/docs/mesa/03-results.md` — measured results (Parts 1-4: 8B/7B/Rev1/sweep;
  **Part 5: Phase 2 Hybrid** — 8B Phase 6 verification + 70B both-AWQ A/B/C
  comparison + ongoing 9D effect).
- `ssd/docs/quantization/{01-plan,02-impl-issues,03-final-report}.md` —
  AWQ Marlin path (v2) + 34B/70B + draft AWQ measurements (legacy v1 torchao
  history kept for reference).

The four legacy root-level files (`MESA-PHASE2-HYBRID-{IMPLEMENTATION-PLAN,
ISSUE,REPORT,FINAL-REPORT}.md`) have been folded into the docs above and
removed.

## Common commands

### SSD engine (work under `ssd/`)

Always `source ssd/env.sh` first — it sets `SSD_HF_CACHE`,
`SSD_TARGET_MODEL` (default `layerskip-llama3-8B`), `SSD_DRAFT_MODEL`
(default `Llama-3.2-1B-Instruct`), `SSD_CUDA_ARCH=8.6` (RTX 3090 box),
and `SSD_DATASET_DIR`. Paths are resolved in `ssd/ssd/paths.py`, which is
imported at the top of `llm_engine.py` before FlashInfer so
`TORCH_CUDA_ARCH_LIST` is set early.

Benchmarks run from inside `bench/` and use `python -O` (debug assertions
off is load-bearing for perf):

```bash
cd ssd/bench
# AR baseline
python -O bench.py --llama --size 8 --gpus 2 --b 1 --temp 0 --numseqs 128 --output_len 512 --all
# Sync spec decode
python -O bench.py --llama --size 8 --spec --k 6 --gpus 2 ...
# Async spec decode (SSD): --async requires 1 extra GPU for the draft process
python -O bench.py --llama --size 8 --spec --async --k 7 --f 3 --gpus 3 ...
# MESA-SSD legacy (two-pass): layered on top of async SD
python -O bench.py ... --spec --async --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 1
# MESA Phase 2 hybrid (current default once --mesa is set with K1/K2):
python -O bench.py ... --spec --async --mesa --k 5 --mesa_phase1_k 3 --mesa_phase2_k 2 --mesa_exit_layer 21
# AWQ W4A16 target (artifact loaded from SSD-native prefix):
python -O bench.py ... --quant_awq --quant_awq_artifact /path/to/awq_artifacts/foo/autoawq_tp4 --quant_group_size 128
# AWQ for the draft as well (Llama-family non-EAGLE, tp=1 only):
python -O bench.py ... --quant_awq_draft --quant_awq_draft_artifact /path/to/draft_awq_artifact
```

Sweep harnesses pin GPU slots and run jobs concurrently:

- `bench/run_mesa_sweep.sh` — 8B target, MESA exit_layer × dfo × phase1/phase2 split.
- `bench/run_34b_sweep.sh` — CodeLlama-34B + TinyLlama, TP=4 target.
- `bench/run_hybrid_sweep_70b.sh` — layerskip-llama2-70B (AWQ TP=4) +
  TinyLlama-1.1B; the canonical Phase 2 hybrid sweep.
- `bench/run_ar_eagle.sh` — AR + EAGLE-3 baselines for comparison.
- `bench/extract_sweep_metrics.py` — post-process sweep result jsons.

Tests:
- `ssd/tests/test_awq_load_validation.py` — CPU-only AWQ artifact I/O +
  schema/role validation (run with `python -m unittest tests.test_awq_load_validation` from `ssd/`).
- `ssd/bench/test_ssd.py`, `ssd/bench/test_llama2.py` — engine smoke tests
  (no pytest config; invoke files directly with `python`).
- `ssd/ssd/utils/async_helpers/tests.py` — async helper unit tests.

### AWQ calibration (separate env)

```bash
/home/chokwans99/anaconda3/envs/awq-quant/bin/python \
  scripts/awq_calibrate_autoawq.py \
  --model /data2/.../layerskip-llama2-70B \
  --out   /data2/chokwans99/awq_calibrated/layerskip_llama2_70b \
  --w-bit 4 --group-size 128 --version GEMM
# then import to SSD-native artifact:
python scripts/awq_import.py --mode autoawq --in <out> --prefix <ssd_artifact_prefix>
```

### Top-level analysis (PSD env)

```bash
bash run_correction.sh     # correction_analysis.py on Llama-70B + Qwen3-32B
bash run_ee_verify.sh      # ee_verify_analysis.py on Llama-70B
```

These run `conda run -n PSD python ...`. Outputs go to
`/home/chokwans99/Parallel_SD/results` (hard-coded in the shell scripts).
Plots are generated by the `plot_*.py` files.

### Cleanup between runs

Async SD spawns multi-process workers (`torch.multiprocessing` spawn
context). If a run crashes or you Ctrl-C, stale workers hold GPU memory:

```bash
pkill -9 -f "bench.py|vllm.entrypoints|sglang.launch_server|VLLM::|sglang::"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

## SSD engine architecture

Entry point is `ssd.LLM` → `LLMEngine` (`ssd/engine/llm_engine.py`). The
engine spawns a constellation of processes that rendezvous through
`torch.multiprocessing` events and NCCL:

- **TP target workers**: one `ModelRunner` per TP rank (`num_tp_gpus =
  num_gpus` for sync, `num_gpus - 1` for async; the extra GPU hosts the
  draft). Rank 0 runs in the main process, others in child processes.
- **DraftRunner**: owns the small draft model and the tree cache. In
  async mode it runs on a dedicated GPU in its own process and
  anticipates verification outcomes in parallel with target verify.
- **Speculator** (`speculator_sync.py` / `speculator_async.py`): the
  coordinator. Produces tree-shaped drafts, requests target verify,
  reconciles accepted suffix.
- **Verifier**: the per-step kernel that folds the draft tree into a
  single target forward pass (uses custom attention masks built in
  `engine/helpers/mask_helpers.py` + `tree_layout.py`).
- **Step** classes (`AutoRegressiveStep`, `SpecDecodeStep`, …): the
  state machine for one generation step, invoked from `llm_engine.step()`.
- **Scheduler / BlockManager / Sequence**: vLLM-style paged KV cache,
  prefix caching, admission of new sequences.
- **CUDA graphs**: captured in `engine/helpers/cudagraph_helpers.py` for
  decode, verify, and tree-decode; replayed each step unless
  `config.enforce_eager=True`. With Phase 2 hybrid enabled, **8 CG
  families** are captured: `glue_long`/`short`, `phase1_long`/`short`,
  `phase2_hybrid_long`/`short`, `verify_long`/`short`.

Layers under `ssd/ssd/layers/` are TP-aware replacements for Linear,
LayerNorm, Rope, Sampler, Attention. They are the integration boundary
for AWQ quantization (Linear forward dispatches to `awq_matmul` when an
`AwqQuantState` is attached).

### Config invariants (from `ssd/config.py::Config.__post_init__`)

Many invariants are enforced in `__post_init__`; trust them instead of
re-checking at call sites:

- `num_gpus <= 8` — single-node only.
- `kvcache_block_size >= 2*k + 2` — required for tree verify.
- `draft_async=True` requires `num_gpus > 1` (one GPU reserved for draft).
- Target and draft must share `infer_model_family()`.
- Eagle draft's `rope_theta` / `max_position_embeddings` are overridden
  to match target — do not re-override at the model level.
- **MESA-SSD** requires: `draft_async=True`, `speculate=True`, Llama model,
  `jit_speculate=True`, CUDA graphs on (`enforce_eager=False`),
  `max_num_seqs=1` (Policy A uses `accept_probs[0]` as a single `h_i`
  for the whole batch). `mesa_exit_layer` defaults to `2*L//3`,
  `mesa_draft_fan_out` to `async_fan_out//2`. `mesa_proxy_top_k` is
  auto-raised so the proxy can always cover the worst-case fan-out
  without fallback.
- **MESA Phase 2 hybrid** is gated by both `mesa_phase1_k` and
  `mesa_phase2_k` being set; both `None` keeps the legacy two-pass path.
  When set, `mesa_phase1_k + mesa_phase2_k == speculate_k` is enforced.

### MESA-SSD (what this fork adds beyond upstream SSD)

Goal: split draft into stages around a target early-exit. **Phase 1**
runs the draft model `K1` forwards producing draft-sourced seed
sequences (acceptance-preserving SD sampling). **Phase 2** consumes the
target's early-exit proxy distribution (`p_i^E`) for every verify
position, computes approximate accept-prob `α̂_i = min(1, p_i^E / p_i^D)`,
first-reject position distribution `ĥ_i`, and residual correction
tokens `r̂_i(v) ∝ [p_i^E - p_i^D]_+`, then fills the async tree cache
with branches more likely to match the real recovery distribution.

The **Phase 2 hybrid** loop (current default) collapses Phase 2 into a
single batched forward of depth `K2 = K_short = K_long - K1` that
processes both **continuation rows** (extending Phase 1 leaves) and
**proxy-sourced rows** (independent K2-deep sequences from proxy tokens)
in one tree decode pass. The hybrid plan object — `HybridPhase2Plan` in
`ssd/engine/helpers/hybrid_phase2_plan.py` — is allocated once at
engine init sized for the long-bucket worst case; per-step
`begin_step` fills it in-place. `valid_k` per cache row picks
long/short bucket dispatch at runtime, selecting which CG to replay.

Hybrid-related env-var gates (debug / regression):

- `SSD_FORCE_SPLIT_PHASE2=1` — force the legacy split (continuation pass
  + proxy pass) Phase 2 path.
- `SSD_FORCE_EAGER_HYBRID_PHASE2=1` — run the hybrid loop eagerly
  (skip CG replay).
- `SSD_HYBRID_PARITY=1` — run BOTH hybrid and split paths, compare,
  log first-divergence drift (debug only).
- `SSD_TRACE_BUCKET=1` — print per-step long/short bucket selection.

### Quantization (AWQ W4A16, current public path)

`ssd/quant/` is a full package built on **sgl-kernel** (`gptq_marlin_gemm`,
`awq_marlin_repack`, `scalar_types.uint4`). Public surface (`ssd.quant`):

- `QuantConfig` — structured config.
- `AwqQuantState` — rank-local packed weight + scales + zeros + metadata.
- `awq_matmul(x, state, bias=None)` — Marlin matmul wrapper, fp16/bf16
  activations, graph-safe.
- `quant_init_context()` — meta-device linear `__init__` so quantized
  weights skip dense allocation.
- `load_awq_artifact` / `save_awq_artifact` — SSD-native artifact I/O
  with role-aware (target vs. draft) validation.

External AutoAWQ checkpoints are repacked into Marlin layout via
`scripts/awq_import.py --mode autoawq` (depends on offline calibration
done in the `awq-quant` env, see above).

Both **target** and **draft** can be AWQ-quantized independently
(`target_quant_enabled` / `draft_quant_enabled`, default backend
`awq_marlin`). Draft quantization is Llama-family non-EAGLE, tp=1 only;
draft `lm_head` and embeddings remain dense (the runner ignores any
opt-in for those — fields exist as no-ops). Target `lm_head` quantization
is opt-in via `target_quant_lm_head` and known to hurt accept rate
(MESA −4–8%p observed); leave default off unless memory-constrained.

The legacy torchao backends (`int4_wo_tile`, `int8_wo`) are kept only as
internal fallback. Setting them logs `[quant][LEGACY]` and is no longer
driven by any CLI. Both are bf16-activation workflows, so combining them
with a fp16 checkpoint requires `target_quant_force_bf16_runtime=True`,
which promotes the entire runtime (KV cache + graph buffers) to bf16.

## Conventions and gotchas

- **Always run engine code with `python -O`.** There are assertions and
  debug hooks expensive enough to dominate the benchmark.
- **Paths are env-driven.** `ssd/ssd/paths.py` raises at import time if
  `SSD_HF_CACHE` or `SSD_DATASET_DIR` are unset — `env.sh` is the source
  of truth for local dev.
- **Separate envs for baselines and AWQ calibration.** Do not
  `pip install sglang`, `vllm`, or AutoAWQ into the `ssd` env;
  FlashInfer / torch / transformers versions collide. Use the conda envs
  `sglang`, `vllm016`, `awq-quant` documented above.
- **Triton / Inductor caches on local disk.** If you see `OSError:
  [Errno 116] Stale file handle`, set `TRITON_CACHE_DIR` and
  `TORCHINDUCTOR_CACHE_DIR` to local storage (existing scripts assume
  `/scratch/$USER/...`).
- **Intermediate artifacts are gitignored** under `ssd/tmp/`,
  `ssd/bench/tmp*/`, `ssd/bench/results/`, `wandb*/`. Plots go to
  `ssd/sandbox/` / `ssd/bench/` during experimentation and to `report/`
  when reported.
- **Top-level and `ssd/` scripts measure different things.** The
  top-level `*_analysis.py` files never touch the SSD engine — they
  load HF models directly to measure distribution-level feasibility of
  the MESA proxy. The SSD engine only exercises the integrated
  inference path.
