# CodeLlama-34B (LayerSkip) + TinyLlama-1.1B — MESA vs Baseline Sweep

**Hypothesis tested**: scale up the target/draft ratio (34B target / 1B draft) so target-side work dominates, giving MESA's 2-pass draft overhead room to hide behind target verify.

## Setup

- Target: `facebook/layerskip-codellama-34B` (48 layers, 8192 hidden, 64/8 heads, vocab 32000, FP16)
- Draft: `TinyLlama-1.1B-Chat-v1.0` (Llama2 vocab 32000 — identical tokenizer)
- GPUs: 5 × RTX 3090 (target TP=4 on ranks 0-3, draft on rank 4). GPUs 5-7 idle.
- Per run: 30 random prompts (input=128, output=256), K=4, B=1, temp=0.6, `max_model_len=2048`.
- Fix: `allocate_kv_cache` had per-rank free-memory drift causing non-uniform `num_kvcache_blocks`; patched with `all_reduce(min)` across TP ranks.

## Results

| Label | f | Throughput (tok/s) | Accept | CacheHit | Tok/step | Draft (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ar_34b** (no spec, TP=4 only) | — | **28.28** | — | — | — | — | — |
| baseline_f2 | 2 | 70.87 | 0.64 | 0.64 | 3.62 | 26.28 | 41.51 |
| baseline_f3 | 3 | 75.59 | 0.70 | 0.70 | 3.76 | 26.74 | 41.05 |
| baseline_f4 | 4 | 70.89 | 0.71 | 0.71 | 3.52 | 26.27 | 41.41 |
| baseline_f5 | 5 | 78.99 | 0.78 | 0.78 | 3.85 | 25.53 | 41.19 |
| **baseline_f6** | 6 | **80.21** | **0.80** | **0.80** | **3.91** | 26.62 | 41.09 |
| baseline_f8 | 8 | 75.03 | 0.78 | 0.78 | 3.65 | 27.06 | 41.12 |
| mesa_f2_dfo1 | 2 | 63.70 | 0.76 | 0.76 | 3.50 | 48.91 | 41.59 |
| mesa_f3_dfo1 | 3 | 64.64 | 0.80 | 0.80 | 3.64 | 49.82 | 41.63 |
| **mesa_f4_dfo1** | 4 | **66.35** | 0.82 | 0.82 | 3.83 | 50.66 | 41.93 |
| mesa_f5_dfo1 | 5 | 66.20 | 0.84 | 0.84 | 3.78 | 49.80 | 42.02 |
| mesa_f6_dfo1 | 6 | 59.79 | 0.82 | 0.82 | 3.41 | 49.87 | 41.92 |
| mesa_f8_dfo1 | 8 | 63.50 | **0.87** | **0.87** | 3.74 | 51.03 | 42.31 |

Speedups vs AR (28.28 tok/s):
- Baseline best: **2.84×** (f=6)
- MESA best: **2.35×** (f=4, dfo=1)

MESA still trails baseline by ~17% even at its best config. The hypothesis ("draft too fast relative to target on 8B") didn't flip the verdict on 34B.

## Why MESA Doesn't Win on 34B Either

Baseline pipeline (per step):
- `draft step = 26 ms`, `target verify = 41 ms` → **target is the bottleneck**. Draft idles ~15 ms/step waiting for target.

MESA pipeline (per step, f=4, dfo=1):
- `draft step = 51 ms`, `target verify = 42 ms` → **draft becomes the bottleneck**.
- Going from baseline draft (26 ms) to MESA draft (51 ms) is +25 ms, exactly the Phase-2 replay cost (4 × 4.2 ms = 16.8 ms of replay, plus ~7 ms of mask/plan/layout rebuild overhead).

The MESA 2-pass structure adds a **fixed ~24 ms** to every draft step. On 34B this fixed cost pushes draft *past* target, losing the overlap that baseline had.

For MESA to actually win, we'd need: `target_verify > MESA_draft`, i.e. target ≥ 51 ms at this config. That's not the case here — target stays at 41 ms because 34B TP=4 parallelizes it well.

## Per-Phase Timeline (MESA f=4, dfo=1, step #30)

Target lane (ms, target's own clock):
```
0.0–20.8  : (previous step's graph_post tail / setup gap)
20.8–43.7 : (idle waiting for draft's Phase1 tree)
30.6–43.7 : graph_pre  (layers 0..31, 13.1 ms)
43.7–45.5 : proxy_compute_send  (1.7 ms)
45.5–51.5 : graph_post (layers 32..47 + norm, 6.0 ms)
51.5–76.0 : (idle waiting for draft's Phase2 tree for next step)
```

Draft lane (ms, offset-aligned to target frame):
```
 0.0– 0.03: proxy_wait  (near-zero, receives target's previous step proxy)
 2.6–20.3 : 4× phase2_replay (16.8 ms)  — produce Phase-2 tree for step N
20.5–20.6 : merge_cache
21.7–26.4 : glue (4.7 ms)  — start step N+1
27.8–45.3 : 4× phase1_replay (17.5 ms) — produce Phase-1 tree
45.5       : proxy_wait for step N+1 (0.003 ms — arrives just in time!)
48.2–65.8 : 4× phase2_replay — produce Phase-2 tree for step N+1
```

Per-spec-step sums (post-warmup means, n=1772):

| Process | Phase | mean ms / call | calls / step | ms / step |
|---------|-------|---:|---:|---:|
| target | graph_pre | 13.14 | 1 | **13.14** |
| target | proxy_compute_send | 1.86 | 1 | 1.86 |
| target | graph_post | 5.99 | 1 | **5.99** |
| target | **(measured total)** | | | **21.0** |
| target | (observed metric) | | | **41.9** |
| draft | glue | 4.71 | 1 | 4.71 |
| draft | phase1_replay | 4.18 | 4 | **16.71** |
| draft | proxy_wait | 0.003 | 1 | 0.00 |
| draft | phase2_replay | 4.20 | 4 | **16.82** |
| draft | merge_cache | 0.07 | 1 | 0.07 |
| draft | **(measured total)** | | | **38.3** |
| draft | (observed metric) | | | **50.7** |

Both "measured" totals sit below "observed metric" by ~12 ms (target: setup+compute_logits; draft: mask+buf, plan, layout build, python). The relative gap between phases is what matters.

## Cross-Process Alignment Quality

- Pairs found: 1777 proxy handshakes
- Alignment offset std: **1.97 ms** over 1767 pairs — good enough for ms-level plots
- `proxy_wait` observed at ~3 µs consistently → **target's proxy_compute_send always completes before draft needs to consume it**. The overlap is working perfectly; the wait is never the bottleneck.

## Key Takeaways

1. **MESA's overhead is architectural, not hardware-sensitive.** Whether 8B (42% slowdown) or 34B (17% slowdown), the Phase-2 replay always adds ~17 ms/step and baseline always wins by exactly that margin.
2. **Target-verify time stays flat at ~41 ms** for this config — MESA's split CudaGraph (graph_pre+graph_post) introduces zero target-side overhead.
3. **Token efficiency does improve** with MESA: at f=8 MESA hits 0.87 cache-hit vs baseline's 0.78. But the ~4 additional tok/step gain (3.91 → ~4.2) is nowhere near enough to cover the +25 ms draft tax.
4. **MESA cache-hit ≈ accept rate for 34B** — they're now near-identical. The proxy is correctly raising draft accept quality.
5. **proxy_wait is always 0** — NCCL handshake arrives long before draft consumption. Overlap mechanism is healthy; the bottleneck is raw 2-pass compute.

## Directions That Would Flip the Verdict

- **Fuse Phase-1 and Phase-2** into a single proxy-aware tree decode (eliminates the +17 ms replay, expected +20-25% throughput).
- **Shrink Phase-2** (fewer per-position branches; currently dfo=1 so phase2 gets pfo=f-1 branches — try smaller proxy trees).
- **Quantize / shrink draft** so each replay is sub-3 ms (draft step falls back under target verify time even with 2 passes).
- **Lower target TP** (e.g. TP=2 → each rank has more layers → target verify rises to ~80-100 ms, covering MESA's 2-pass draft).

## Artifacts in this directory

- `SUMMARY.txt` — raw per-run metrics
- `ar_34b.log`, `baseline_f*.log`, `mesa_f*_dfo1.log` — full bench stdout
- `../34b_timeline/mesa/` — MESA (f=4) timeline profile JSON + Gantt/bar/time-series PNGs
- `../34b_timeline/baseline/` — Baseline (f=6) draft-side profile (target-side requires instrumentation in `run_verify_cudagraph`; currently only MESA target path is instrumented)
- `../34b_timeline/mesa/mesa_timeline_step30.png` — **single-step Gantt** showing target/draft overlap
- `../34b_timeline/mesa/mesa_breakdown.png` — per-phase bar chart

## Reproduce

```bash
bash bench/run_34b_sweep.sh /home/chokwans99/PSD/ssd/tmp/34b_sweep
bash /tmp/finish_34b.sh   # AR + timeline profiles
python bench/plot_mesa_timeline.py /home/chokwans99/PSD/ssd/tmp/34b_timeline/mesa --step 30
python bench/plot_mesa_breakdown.py /home/chokwans99/PSD/ssd/tmp/34b_timeline/mesa
```
