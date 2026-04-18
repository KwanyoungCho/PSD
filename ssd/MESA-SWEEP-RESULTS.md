# MESA-SSD Parameter Sweep (Clean-GPU Rerun)

## Environment

- **Target**: LayerSkip-Llama3-8B (32 layers) [MESA / SSD baselines]
- **Target (EAGLE track)**: Llama-3.1-8B-Instruct
- **Draft (SSD/MESA)**: Llama-3.2-1B-Instruct
- **Draft (EAGLE)**: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
- **GPUs**: 8× RTX 3090 (CUDA_VISIBLE_DEVICES isolated, TP=2 per run, 4 runs in parallel across slots)
- **Previous run was contaminated** by another user's job on GPUs 1/4. Rerun is on verified-empty GPUs.
- **Prompts**: 300 random token sequences (input_len=128), temp=0.6, K=4, output_len=256, B=1
- **Speculation knobs**: --async --spec --k 4, baselines vary --f ∈ {2,3,4,5} to match MESA's total draft budget (MQ_LEN = f·(K+1))

## Phase A — Baseline SSD (matched budget)

| Config | f (budget 5f) | Throughput | Accept | CacheHit | Tok/Step | Draft (ms) | Verify (ms) |
|--------|---------------|------------|--------|----------|----------|------------|-------------|
| baseline_f2 | 2 (MQ=10) | 139.57 | 0.81 | 0.82 | 4.23 | 24.52 | 22.96 |
| **baseline_f3** | **3 (MQ=15)** | **142.02** | **0.82** | 0.85 | **4.29** | 24.32 | 22.97 |
| baseline_f4 | 4 (MQ=20) | 133.78 | 0.82 | **0.86** | 4.28 | 25.60 | 22.93 |
| baseline_f5 | 5 (MQ=25) | 131.45 | 0.81 | **0.86** | 4.22 | 25.92 | 23.01 |

Baseline SSD peaks at f=3 (the existing default). Larger trees buy a tiny amount of cache-hit rate but the extra tree-decode compute reverses the gain. Accept rate is essentially flat — the draft is already close to saturation at f=3.

## Phase A — MESA (exit_layer=21 fixed, sweep f × draft_fan_out split)

Total budget = f · (K+1). Split notation (dfo, pfo) = (Phase-1 per-position branches, Phase-2 per-position branches), dfo + pfo = f.

| Config | f | Split (dfo, pfo) | Throughput | Accept | CacheHit | Tok/Step | Draft (ms) | Verify (ms) |
|--------|---|------------------|------------|--------|----------|----------|------------|-------------|
| mesa_f2_dfo1 | 2 | (1, 1) | 88.58 | 0.83 | 0.87 | 4.31 | 43.71 | 24.79 |
| **mesa_f3_dfo1** | **3** | **(1, 2)** | **87.54** | 0.81 | 0.88 | 4.24 | 43.40 | 24.75 |
| mesa_f3_dfo2 | 3 | (2, 1) | 85.29 | 0.79 | 0.87 | 4.15 | 43.74 | 24.64 |
| mesa_f4_dfo1 | 4 | (1, 3) | 85.29 | 0.79 | 0.88 | 4.18 | 43.93 | 24.72 |
| mesa_f4_dfo2 | 4 | (2, 2) | 85.47 | 0.79 | 0.88 | 4.18 | 43.84 | 24.77 |
| mesa_f4_dfo3 | 4 | (3, 1) | 87.10 | 0.82 | 0.88 | 4.26 | 44.02 | 24.65 |
| mesa_f5_dfo1 | 5 | (1, 4) | 82.91 | 0.80 | **0.90** | 4.21 | 45.17 | 24.89 |
| mesa_f5_dfo2 | 5 | (2, 3) | 86.76 | 0.81 | 0.89 | 4.26 | 43.99 | 24.73 |
| mesa_f5_dfo3 | 5 | (3, 2) | 86.37 | 0.81 | 0.89 | 4.24 | 44.11 | 24.75 |
| mesa_f5_dfo4 | 5 | (4, 1) | 82.19 | 0.80 | 0.88 | 4.18 | 45.93 | 25.58 |

All MESA configurations sit in a narrow band (82–89 tok/s). **Every MESA config is 35–42% slower than baseline_f3**, *despite* every MESA config having a higher cache-hit rate. The token-efficiency gain is real but small (+0.02–0.05 CH, ≈-0.02 accept in a few), whereas the wall-clock draft-step nearly *doubles* (43–46 ms vs. baseline's 24–26 ms). Target verify is also ~2 ms slower than baseline, so the split-CudaGraph (graph_pre + proxy + graph_post) does not actually save target-side time on these GPUs.

## Phase B — exit_layer sweep (at f=3, dfo=1)

| Config | exit_layer (% of L=32) | Throughput | Accept | CacheHit | Tok/Step | Draft (ms) | Verify (ms) |
|--------|------------------------|------------|--------|----------|----------|------------|-------------|
| mesa_f3_dfo1_exit10 | 10 (31%) | 85.00 | 0.79 | 0.85 | 4.15 | 43.83 | 24.75 |
| **mesa_f3_dfo1_exit16** | **16 (50%)** | **88.41** | **0.81** | 0.87 | **4.26** | 43.18 | 24.71 |
| mesa_f3_dfo1 (=21) | 21 (66%) | 87.54 | 0.81 | 0.88 | 4.24 | 43.40 | 24.75 |
| mesa_f3_dfo1_exit26 | 26 (81%) | 86.54 | 0.80 | 0.89 | 4.21 | 43.64 | 24.72 |

With clean GPUs the exit-layer effect is shallower than the previous (noisy) sweep suggested — all four points land within ~3 tok/s of each other. The optimum shifts slightly to **exit=16 (50%)**, which reports the best accept rate *and* tok/step; exit=21 keeps the best cache-hit rate. Split-timing intuition: earlier exit → proxy arrives sooner → Phase-2 starts sooner, but the early-exit logits quality is weaker and draft budgeting becomes less informative. Later exit → stronger proxy but Phase-2 is gated later.

## Phase C — AR (no speculation) and EAGLE (different target)

For reference, AR runs execute the target model alone (no draft, no tree):

| Config | Target | Throughput | Time (s) |
|--------|--------|-----------:|---------:|
| ar_layerskip | LayerSkip-Llama3-8B | 74.39 | 1032.38 |
| ar_llama31 | Llama-3.1-8B-Instruct | *(pending)* | *pending* |
| eagle_f3_k4 | Llama-3.1-8B-Instruct + EAGLE-3 | **not run** | OOM at 23.4 GB/GPU with default/max_model_len=2048; likely a TP-split issue with EAGLE's 1-layer draft on 24 GB cards. Needs a dedicated reproducer before retrying. |

AR throughput (74.39 tok/s) is the spec-decoding floor. Baseline SSD speeds it up ~1.9× (74 → 142); MESA speeds it up only ~1.2× (74 → 87). **MESA is still faster than AR, but does not catch up to baseline SSD on this config / hardware.**

## Comparison Summary

| Mode | Target | Best config | Throughput (tok/s) | vs AR (LayerSkip) | vs Baseline_f3 |
|------|--------|-------------|-------------------:|-----------------:|---------------:|
| AR | LayerSkip-Llama3-8B | — | 74.39 | 1.00× | 0.52× |
| Baseline SSD | LayerSkip-Llama3-8B | f=3 | **142.02** | **1.91×** | 1.00× |
| MESA SSD | LayerSkip-Llama3-8B | f=3, dfo=1, exit=16 | 88.41 | 1.19× | **0.62×** |
| EAGLE | Llama-3.1-8B | — | *OOM (not run)* | — | — |

## Root-Cause Analysis

1. **Token efficiency is real but small.** MESA consistently raises cache hit by +0.02–0.05 and pushes tok/step up by up to +0.1 in the good configs. This confirms the proxy-driven selection works — it just doesn't move the throughput dial enough to beat the structural cost.

2. **Draft step is the bottleneck, and it is ~2× baseline.** Baseline draft step is 24–26 ms. MESA draft step is 43–46 ms, across *every* config. The gap is dominated by the 2-pass structure:
   - Two CudaGraph replays (draft layout + proxy layout) per step instead of one.
   - Two FlashInfer `wrapper.plan()` calls per step (draft + proxy layouts).
   - Per-layout mask precompute.
   - Glue compute + proxy-cache merge (Policy A dynamic fan_out_list) adds ~2-4 ms over baseline's single-pass glue.
   That roughly matches the earlier budget estimate of ~37 ms of structural overhead on top of the raw decode work.

3. **Target verify is not actually faster with split CudaGraph.** Baseline verify is 23 ms; MESA (graph_pre + proxy + graph_post) is 24.7–25.6 ms. The overhead of splitting the CudaGraph in two and handshaking through a proxy tensor eats the savings from "Phase-2 starts earlier" on this size of model. So the motivating benefit for split CudaGraph doesn't materialize here.

4. **Sweep-knob sensitivity is low.** Over the 10-config f×split grid at exit=21, MESA throughput ranges only 82.19–88.58 tok/s (±4%). Exit-layer sweep is even flatter (85.00–88.41). The space is roughly convex with a shallow optimum at (f=3, dfo=1, exit≈16–21), but no knob choice gets MESA out of the -35% band vs baseline.

5. **The previous "MESA beats baseline" result was a GPU-contention artifact.** On shared GPUs baseline's target verify had inflated to 65 ms; on clean GPUs it drops to 23 ms and the gap reverses.

## What this means for MESA as designed

On this model size (8 B, 32 layers) and hardware (RTX 3090, TP=2, B=1), the structural overhead of the 2-pass CudaGraph + dual-layout FlashInfer plan outweighs the ~5–10% token-efficiency gain that the proxy delivers. To change this verdict MESA would need at least one of:

- a dramatically cheaper Phase-2 (e.g., collapse to 1 CudaGraph keyed on a runtime layout descriptor instead of recompiling/re-planning per step),
- a larger target (70 B) where the proxy-driven Phase-1→Phase-2 pipeline can actually hide latency, or
- a regime where baseline's draft is *not* already near-saturating accept rate, so the proxy's cache-hit improvement actually compounds.

Until one of those holds, baseline SSD at f=3 remains the right default.

## Notes

- `mesa_f2_dfo1` and `mesa_f3_dfo2` failed once with `DistStoreError` during rapid slot reuse (multiprocessing-spawn race on port reuse). Reruns on the same hardware produced the numbers in the table above; the failure was not algorithmic.
- Summary regex in `bench/run_mesa_sweep.sh` was also fixed to capture "Total Throughput:" correctly (previously it captured "76800tok" = total token count).
