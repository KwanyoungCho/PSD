# Final MESA vs Baseline SSD Experiment (Real Datasets)

## Setup

| Item | Value |
|------|-------|
| **Target** | `facebook/layerskip-codellama-34B` (48 layers, 8192 hidden, 64/8 heads) |
| **Draft** | `TinyLlama-1.1B-Chat-v1.0` (Llama2 vocab, 22 layers) |
| **GPUs** | 5 × RTX 3090 — Target TP=4 (rank 0-3), Draft on rank 4. `CUDA_VISIBLE_DEVICES=0,1,2,3,4` |
| **Prompts** | 200 = 50 × {humaneval, alpaca, gsm8k, ultrafeedback} (`--all --numseqs 50`) |
| **Tokenization** | by target tokenizer (Llama2 SentencePiece), padded/truncated to input_len=128 |
| **Generation** | `output_len=256`, `temp=0.6` (stochastic — MESA residual effective), `B=1`, `max_model_len=2048` |
| **Speculation knobs** | `--k 6` |
| **Baseline SSD fan_out** | front-loaded geometric `--flh 5 4 4 3 2 2 1` → MQ_LEN=21 |
| **MESA fan_out** | uniform `--f 3 --mesa_draft_fan_out 1` (Phase-1 fo=1, Phase-2 fo=2) → MQ_LEN=21 matched |
| **MESA exit layer** | 24 (1/2 of 48) and 32 (2/3 of 48) |
| **Dataset path** | `/data2/chokwans99/datasets/` (jsonl, 50 rows each) |
| **Profiling** | `SSD_PROFILE_MESA=1` zero-sync CUDA events |

## Reproduction

```bash
bash ssd/tmp/final_exp/run_all.sh
# Uses env:
#   SSD_HF_CACHE=/data2/chokwans99/models
#   SSD_DATASET_DIR=/data2/chokwans99/datasets
#   TORCH_CUDA_ARCH_LIST=8.6  (RTX 3090)
```

## Main Results

| Config | Throughput (tok/s) | Speedup vs AR | Accept | **Cache hit** | Tok/Step | Target verify (ms) | Draft step (ms) |
|--------|-------------------:|--------------:|:------:|:-------------:|:--------:|-------------------:|----------------:|
| **AR** (target only, TP=4, no spec) | **28.52** | 1.00× | — | — | — | — | — |
| **Baseline SSD** (K=6, geo `[5,4,4,3,2,2,1]`) | **66.44** | **2.33×** | 0.46 | **0.66** | 3.75 | 45.4 | 41.0 |
| **MESA exit=24 (1/2)** | **55.09** | 1.93× | 0.48 | **0.80** | 3.88 | 46.4 | 64.6 |
| **MESA exit=32 (2/3)** | **56.94** | 2.00× | 0.48 | **0.84** | 3.91 | 46.3 | 62.7 |

### Key observations
- **Baseline SSD wins** at 66.44 tok/s. MESA configs are 14-17% slower.
- **MESA lifts cache hit dramatically** (0.66 → 0.84, +18pp) and accept rate (0.46 → 0.48, +2pp). Token efficiency per step rises ~4%.
- **But the Phase-2 cost adds ~22 ms to draft step** (41.0 → 62.7 ms), which outweighs the efficiency gain.
- **Exit=32 (2/3) > Exit=24 (1/2)** on real datasets: deeper exit → better proxy quality → +4pp cache hit + slightly faster overall. Opposite of the random-prompt finding where 1/2 was slightly better.

## Per-Phase Breakdown (post-warmup means, per spec step)

### Baseline SSD (K=6, geo fanout, MQ=21)

**Target (per step ≈ 53 ms):**
| phase | ms |
|-------|---:|
| verify_replay | 41.78 |
| target_spec_wait (draft 기다림) | 8.99 |
| verify_sample_accept | 2.24 |
| target_postprocess | 0.04 |
| **sum** | **53.05** |

**Draft (per step ≈ 53 ms):**
| phase | ms | 횟수/step |
|-------|---:|:---:|
| glue (incl. draft_glue_replay 3.50) | 4.02 | 1 |
| draft_recv_cmd (target cmd 기다림) | 10.85 | 1 |
| hit_cache_respond (cache lookup + JIT speculate on miss) | 7.89 | 1 |
| tree_prep | 0.44 × 6 = 2.65 | 6 |
| tree_replay | 4.59 × 6 = **27.56** | 6 |
| draft_send_response | 0.16 | 1 |
| **sum** | **53.13** | |

- Target/draft balanced at ~53 ms — good pipeline.
- Tree budget MQ=21 split across K=6 iterations, geometric means first iter has 5-branch tree, last has 1.

### MESA exit=24 (1/2)

**Target (per step ≈ 67 ms):**
| phase | ms |
|-------|---:|
| verify_setup | 0.23 |
| **graph_pre** (layers 0-23, 24/48 = 50%) | **22.59** |
| exit_logits (vocab proj on exit_h) | 0.44 |
| proxy_compute_send (h_i, r_i, topk, isend) | 0.75 |
| **graph_post** (layers 24-47) | **18.97** |
| final_logits | 0.39 |
| verify_sample_accept | 2.26 |
| target_spec_wait (draft 기다림) | **21.76** |
| target_postprocess | 0.04 |
| **sum** | **67.43** |

**Draft (per step ≈ 66 ms):**
| phase | ms | 횟수 |
|-------|---:|:---:|
| glue | 4.61 | 1 |
| draft_recv_cmd | 1.71 | 1 |
| hit_cache_respond | 5.01 | 1 |
| phase1_build (select + args) | 0.19 | 1 |
| phase1_prep (mask+plan) | 0.47 × 6 = 2.79 | 6 |
| **phase1_replay** (MQ=6 per iter) | 3.84 × 6 = **23.06** | 6 |
| proxy_wait (target 기다림) | 0.05 | 1 |
| phase2_build (Policy A layout+select) | 1.68 | 1 |
| phase2_prep | 0.44 × 6 = 2.66 | 6 |
| **phase2_replay** (MQ=12 per iter) | 4.23 × 6 = **25.35** | 6 |
| merge_cache | 0.04 | 1 |
| draft_send_response | 0.18 | 1 |
| **sum** | **67.38** | |

### MESA exit=32 (2/3)

**Target (per step ≈ 66 ms):**
| phase | ms |
|-------|---:|
| verify_setup | 0.23 |
| **graph_pre** (layers 0-31, 32/48 = 67%) | **29.39** |
| exit_logits | 0.44 |
| proxy_compute_send | 0.67 |
| **graph_post** (layers 32-47, 33%) | **12.27** |
| final_logits | 0.42 |
| verify_sample_accept | 2.10 |
| target_spec_wait | **20.11** |
| target_postprocess | 0.03 |
| **sum** | **65.66** |

**Draft (per step ≈ 66 ms):** 동일 구조, `proxy_wait` 0.45 ms (exit=24 대비 약간 길어 target이 graph_pre 더 오래 걸리는 만큼 대기).

## Critical Path Analysis

**Baseline bottleneck**: target verify_replay (41.8 ms) dominates. Target-bound.
- Draft tree build finishes at ~t=52 ms. Target still has 9 ms spec_wait to absorb.
- Draft `hit_cache_respond` = 7.9 ms avg includes cache miss → JIT speculate (~15 ms when miss).

**MESA bottleneck**: **draft 2-pass structure** adds ~22 ms vs baseline.
- Phase-2 replay × 6 = 25 ms ≈ exact same order as baseline tree_replay × 6 (27 ms), but it's **additional work** on top of Phase-1.
- Target waits `spec_wait = 20 ms` — target is ready but draft is still producing tree.
- i.e. on 34B target, **draft becomes the bottleneck in MESA**, not target.

## Comparison to Previous (Random Prompts) Result

| Metric | Random (earlier) | **Real Datasets** (this) |
|--------|:----------------:|:-----------------------:|
| Baseline accept | 0.80 | **0.46** ← real prompts harder |
| MESA accept | 0.80 | **0.48** |
| Baseline cache hit | 0.82 | **0.66** |
| MESA cache hit (exit=24) | 0.87 | **0.80** ← **MESA gap larger on real prompts!** |
| MESA cache hit (exit=32) | 0.87 | **0.84** |
| Best MESA exit | 24 (1/2) | **32 (2/3)** ← deeper proxy pays off on real data |

**On real prompts**:
- Accept rate drops for everyone (0.80 → 0.46) because real text is harder than random ids
- Cache hit drops even more for baseline (0.82 → 0.66), but MESA is much more robust (0.87 → 0.84)
- **MESA's proxy-based cache filling is much more valuable when naive speculation often misses** — but the 2-pass cost still exceeds the gain

## Artifacts

```
tmp/final_exp/
├── REPORT.md                       — this file
├── run_all.sh                      — reproducible script
├── SUMMARY.txt                     — 1-line-per-config summary
├── ar/
│   └── run.log
├── baseline/
│   ├── run.log
│   ├── mesa_profile_{draft,target_rank0}_*.json   — zero-sync CUDA events
│   ├── mesa_breakdown.png                         — per-phase bar chart
│   ├── mesa_breakdown_over_time.png
│   ├── mesa_breakdown_summary.csv                 — mean/median/p95 per label
│   └── mesa_per_step_contribution.csv             — per-step × K multiplier
├── mesa_exit24/  (same structure)
└── mesa_exit32/  (same structure)
```

To regenerate plots from JSON:
```bash
python bench/plot_mesa_breakdown.py tmp/final_exp/baseline
python bench/plot_mesa_timeline.py tmp/final_exp/baseline --step 51 --warmup 0
```

## Conclusions

1. **Baseline SSD with geometric fanout (K=6, `[5,4,4,3,2,2,1]`)** is the current sweet spot at **66.4 tok/s** (2.33× AR), with well-balanced target/draft pipeline (~53 ms each).

2. **MESA's proxy mechanism significantly boosts cache quality** (0.66 → 0.84, +18pp on real data). This is the strongest validation we have for the proxy idea: it measurably aligns draft proposals with target's distribution.

3. **But MESA's 2-pass structural overhead (~22 ms Phase-2 replay) exceeds the token efficiency gain (~4%)**. Result: MESA is 14-17% slower than baseline.

4. **Exit layer choice matters (slightly) on real data**: 2/3 (exit=32) > 1/2 (exit=24) by ~3%. Deeper early-exit → better proxy quality → higher cache hit → net win despite graph_pre getting longer.

5. **Residual sampling is active** (`residual = (p_E - p_D).clamp(min=0)` at verifier.py:207) and its benefit shows in the cache hit jump (0.66 → 0.84). But this benefit is still not enough to overcome structural cost.

### Directions to beat baseline (priority order)

1. **Fuse Phase-1 and Phase-2 into a single proxy-aware tree decode**: eliminate the +25 ms Phase-2 replay entirely. Estimated +30% TP recovery.
2. **Non-uniform MESA fan_out**: currently `--f 3` means uniform. Match baseline's geometric `[5,4,4,3,2,2,1]` by splitting each position into (draft, proxy) — would need extending `mesa_draft_fan_out` to a list.
3. **Batch size > 1**: at B=1 the tree is tiny; at B=8+ the fixed Phase-2 overhead amortizes.
4. **Higher temperature** (1.0+): residual correction more impactful.
