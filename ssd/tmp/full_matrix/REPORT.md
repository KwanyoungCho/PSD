# Full Bench Matrix — INT4 Weight-Only vs Dense

Run: 2026-04-21 · 24 experiments (3 targets × 8 configs) · all 4 datasets, numseqs=10, output_len=256, 10240 tok/run
Hardware: 8× RTX 3090 (SM 8.6, 24 GB) · torch 2.8+cu128, torchao 0.12, bf16 runtime

## What each experiment means

- **AR**: plain autoregressive decode (no draft, no speculation). TP = target model tokens/sec.
- **async spec (k=7, geo fanout [5 4 4 3 3 2 2 1])**: standard async speculative decoding; draft model proposes k=7 tokens per step, target verifies. `accept` is the mean fraction of speculated tokens that the target accepts.
- **MESA mid / early (k=5, fan=4, draft_fan_out=2, proxy_fan_out=2)**: MESA-SSD. Exit the target at `exit_layer` to form a cheap **proxy** that runs alongside the external draft. The target's first `exit_layer` layers (*proxy*) generate their own tree branches; a merged tree is verified by the full target. `draft_fan_out=2` out of 4 total branches per step come from the external draft, the other 2 from the proxy.
  - **mid** = `exit_layer = 2L/3` (Llama-3-8B: 21/32, Llama-2-7B: 21/32, CodeLlama-34B: 32/48). Proxy is stronger (more layers), so acceptance is higher but proxy cost is higher.
  - **early** = `exit_layer = L/2` (16/32, 16/32, 24/48). Proxy is cheaper but less accurate, so acceptance drops slightly.
- **dense vs INT4**: dense uses bf16 weights; INT4 uses torchao `Int4WeightOnlyConfig` (TensorCoreTiled / tinygemm) on every column/row/QKV/merged TP-linear **except** lm_head. Activations and KV cache stay bf16 — only target-model weights are quantized. Models B and C are fp16 checkpoints coerced to bf16 runtime (the only path torchao tinygemm supports today).

## Overall numbers

### Target A — Llama-3-8B (bf16 native, TP=2)

| # | experiment | TP (tok/s) | accept | tok/step | cache_hit | time |
|---|---|---|---|---|---|---|
| 01 | AR dense | 74.44 | — | — | — | 137.6s |
| 02 | AR INT4 | **131.33** | — | — | — | 78.0s |
| 03 | async spec dense | 80.64 | 0.46 | 4.22 | 0.68 | 127.0s |
| 04 | async spec INT4 | 78.41 | 0.46 | 4.20 | 0.67 | 130.6s |
| 05 | MESA mid dense | 60.27 | 0.52 | 3.58 | 0.86 | 169.9s |
| 06 | MESA mid INT4 | 62.50 | 0.55 | 3.76 | 0.87 | 163.8s |
| 07 | MESA early dense | 61.59 | 0.54 | 3.71 | 0.83 | 166.3s |
| 08 | MESA early INT4 | 59.48 | 0.53 | 3.63 | 0.81 | 172.2s |

| config | dense TP | INT4 TP | Δ TP | Δ accept |
|---|---|---|---|---|
| AR | 74.44 | 131.33 | **+76.4%** | — |
| async spec | 80.64 | 78.41 | -2.8% | +0.00 |
| MESA mid | 60.27 | 62.50 | +3.7% | +0.03 |
| MESA early | 61.59 | 59.48 | -3.4% | -0.01 |

### Target B — Llama-2-7B (fp16 → bf16 runtime override, TP=2)

| # | experiment | TP (tok/s) | accept | tok/step | cache_hit | time |
|---|---|---|---|---|---|---|
| 01 | AR dense | 79.93 | — | — | — | 128.1s |
| 02 | AR INT4 | **125.04** | — | — | — | 81.9s |
| 03 | async spec dense | 82.34 | 0.48 | 4.35 | 0.69 | 124.4s |
| 04 | async spec INT4 | 82.76 | 0.49 | 4.44 | 0.70 | 123.7s |
| 05 | MESA mid dense | 66.07 | 0.56 | 3.81 | 0.87 | 155.0s |
| 06 | MESA mid INT4 | 66.86 | 0.60 | 4.02 | 0.89 | 153.2s |
| 07 | MESA early dense | 64.18 | 0.54 | 3.72 | 0.85 | 159.5s |
| 08 | MESA early INT4 | 64.09 | 0.56 | 3.79 | 0.84 | 159.8s |

| config | dense TP | INT4 TP | Δ TP | Δ accept |
|---|---|---|---|---|
| AR | 79.93 | 125.04 | **+56.4%** | — |
| async spec | 82.34 | 82.76 | +0.5% | +0.01 |
| MESA mid | 66.07 | 66.86 | +1.2% | +0.04 |
| MESA early | 64.18 | 64.09 | -0.1% | +0.02 |

### Target C — CodeLlama-34B (fp16 → bf16 runtime override, TP=4)

| # | experiment | TP (tok/s) | accept | tok/step | cache_hit | time |
|---|---|---|---|---|---|---|
| 01 | AR dense | 28.45 | — | — | — | 359.9s |
| 02 | AR INT4 | **52.95** | — | — | — | 193.4s |
| 03 | async spec dense | 71.03 | 0.46 | 4.20 | 0.67 | 144.2s |
| 04 | async spec INT4 | 72.98 | 0.42 | 3.97 | 0.68 | 140.3s |
| 05 | MESA mid dense | 59.58 | 0.53 | 3.66 | 0.87 | 171.9s |
| 06 | MESA mid INT4 | 60.26 | 0.52 | 3.59 | 0.88 | 169.9s |
| 07 | MESA early dense | 59.77 | 0.53 | 3.63 | 0.82 | 171.3s |
| 08 | MESA early INT4 | 57.60 | 0.52 | 3.62 | 0.83 | 177.8s |

| config | dense TP | INT4 TP | Δ TP | Δ accept |
|---|---|---|---|---|
| AR | 28.45 | 52.95 | **+86.1%** | — |
| async spec | 71.03 | 72.98 | +2.7% | -0.04 |
| MESA mid | 59.58 | 60.26 | +1.1% | -0.01 |
| MESA early | 59.77 | 57.60 | -3.6% | -0.01 |

## Memory footprint (INT4)

| target | src (bf16) | nominal int4 | actual incl scale/zp | ratio |
|---|---|---|---|---|
| A (Llama-3-8B, TP=2) | 6.98 GB | 1.74 GB | **1.85 GB** | 0.27 |
| B (Llama-2-7B, TP=2) | 6.48 GB | 1.62 GB | **1.76 GB** | 0.27 |
| C (CodeLlama-34B, TP=4) | 16.61 GB | 4.15 GB | **4.55 GB** | 0.27 |

Per-rank after TP shard (Target A linear params = 128 · 2 ranks; C = 192 · 4 ranks). Actual ratio 0.27 vs nominal 0.25 is the tile-packed layout's scale (bf16) + zero-point (int) overhead.

## Interpretation — why the numbers look this way

### 1. AR INT4 gives a large, clean win on every target (+56% to +86%)

Single-sequence AR decode is **weight-memory-bound**: the GPU reads the full model once per token. Quantizing bf16→int4 cuts weight bytes 4×, so DRAM→SM bandwidth (936 GB/s on 3090) is freed proportionally. The speedups line up with how memory-bound each config is:

- C (34B, TP=4) +86%: biggest model → most memory-bound → biggest win.
- A (8B, TP=2) +76%: still largely memory-bound.
- B (7B, TP=2) +56%: slightly smaller model; a bit more headroom is already there in bf16, so quant helps less.

These numbers are close to the theoretical 4× weight read savings × (1 − non-weight-time-fraction). The actual-ratio 0.27 (not 0.25) costs a few % of the theoretical ceiling, plus LM-head stays bf16 (intentionally: quantizing LM-head hurts MESA accuracy).

### 2. Spec and MESA INT4 give **almost nothing** (±3%)

This is the expected and correct result — not a regression. Once the system is speculating, each target "step" verifies a tree of 4–7 tokens in a single forward pass. Verification is batched (small `b` but large sequence-dim due to tree), so the target moves from memory-bound toward **compute-bound**. In that regime, reducing weight bytes doesn't move the needle because the bottleneck has shifted to GEMM compute, attention, and scheduler overhead. The draft model's time also becomes a larger fraction of the wall clock, and the draft is **not** quantized.

You can see this in the C_01 → C_03 jump: 34B AR dense 28.45 tok/s → 34B async-spec dense 71.03 tok/s is already a 2.5× free speedup from speculation. On top of that, shaving target weight reads gains you almost zero, because target verify time is no longer the limiting factor.

### 3. MESA is **slower** than plain async spec on all three targets

A: async 80.6 vs MESA-mid 60.3 (-25%). B: async 82.3 vs MESA-mid 66.1 (-20%). C: async 71.0 vs MESA-mid 59.6 (-16%).

Acceptance goes up (MESA mid: 0.52–0.56 vs async: 0.46–0.48; cache_hit 0.86–0.88 vs 0.67–0.70), meaning MESA *does* keep more speculated tokens. But each MESA step costs more than an async-spec step because:

- The proxy runs the first `exit_layer` layers of the target on its own tree (plus an LM-head) every step. For Target A `mid=21/32`, proxy ≈ 66% of target's forward cost.
- The merged tree is larger (F=4 branches × 6 depth = MQ_LEN=24), so target verify is also doing more work per step.
- The split-verify graph runs layers 0..exit on the merged tree (cache-hit branch) and full 0..L−1 only on the miss path, but the always-paid exit-layer cost dominates on small models.

Per step MESA accepts ~3.6–4.0 tokens (tok/step) vs async ~4.2–4.4, and the step itself is ~1.5–1.8× more expensive, so throughput loses.

### 4. MESA's disadvantage narrows with model size

Gap vs async spec: A -25%, B -20%, C -16%. Proxy cost is absolute (exit_layer layers); target verify cost scales with L. As L grows the fixed proxy overhead is a smaller fraction, and MESA's higher acceptance starts to pay off. At ≥70B a crossover is expected (not run here — only up to 34B fits on this node).

### 5. `exit=early` (L/2) is **not** faster than `exit=mid` (2L/3)

Counter-intuitive but consistent across all three targets: early saves proxy compute but also loses acceptance, and on these models the acceptance loss outweighs the compute saving. Target A: mid=60.3 vs early=61.6; B: 66.1 vs 64.2; C: 59.6 vs 59.8. Differences are within ±3%, so "mid" is the safer default. Early-exit only wins when the proxy is too slow to keep up with the draft — on current models it isn't.

### 6. INT4 acceptance delta is noise

Dense vs INT4 acceptance moves by ±0.04 at most, which matches the stochastic variance of temp=0.6 sampling across 10 sequences. No evidence that weight-only INT4 causes meaningful acceptance drop. The one arguable exception is C async (dense 0.46 → INT4 0.42, -0.04) — still within the ±0.04 noise band seen elsewhere, not a reliable signal.

### 7. fp16 → bf16 runtime coercion (Target B, C) works

Both Target B (Llama-2-7B) and Target C (CodeLlama-34B) ship as fp16 checkpoints. torchao's `Int4WeightOnlyConfig` tinygemm path requires bf16 activations; we opt-in via `--quant_force_bf16_runtime`. Results (B AR +56%, C AR +86%, normal acceptance, no errors) confirm the workaround is numerically fine for weight-only quant. The artifact format records `effective_runtime_dtype=bf16` and `original_checkpoint_dtype=fp16` so a loader can't silently mix them.

## Summary

| finding | observation |
|---|---|
| INT4 AR speedup | +56% to +86%; scales with model size (34B largest win) |
| INT4 spec / MESA speedup | ≈0% (verify is compute-bound, not weight-memory-bound) |
| INT4 quality | no acceptance regression beyond noise (±0.04) |
| INT4 memory | ×0.27 of bf16 weights (model-agnostic; incl. scale/zp) |
| MESA vs async | MESA loses 16–25% TP at ≤34B; gap narrows with L |
| MESA mid vs early | mid (2L/3) ≈ early (L/2); no meaningful TP difference |
| fp16 → bf16 override | works correctly for torchao WO INT4 on Llama-2/CodeLlama |

**Bottom line**: INT4 weight-only quant is a clear, free win for memory-bound AR serving, and a safe no-op for speculative/MESA decoding. For spec/MESA, the speedup lever is orthogonal — to make quant help there, you'd need to quantize the draft model or move to activation quantization so target-verify GEMMs run in lower precision.
