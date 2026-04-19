# MESA Phase-Level Latency Breakdown

**Run**: `--f 3 --mesa --mesa_exit_layer 21 --mesa_draft_fan_out 1`, K=4, B=1, 30 seqs × 256 out, LayerSkip-Llama3-8B TP=2 on RTX 3090 (GPU 0,1).  
**Throughput**: 85.98 tok/s (consistent with 300-seq sweep: 87.54).

## Files

| File | Description |
|------|-------------|
| `mesa_profile_draft.json` | Raw (idx, label, ms) for every draft-process event (19547 entries) |
| `mesa_profile_target_rank0.json` | Same for target rank 0 (5331 entries) |
| `mesa_breakdown_summary.csv` | mean/median/p95 per (proc, label), post 5-event warmup |
| `mesa_per_step_contribution.csv` | Same but multiplied by K=4 for per-replay labels |
| `mesa_breakdown.png` | Grouped bar chart (per-invocation means) |
| `mesa_breakdown_over_time.png` | Scatter of draft-process latency vs event idx |
| `mesa_run.log` | Full bench.py stdout |

## Results (post-warmup, mean ms)

| proc | label | mean | median | p95 | n | mult/step | ms/step |
|------|-------|-----:|-------:|----:|---:|----------:|--------:|
| target | graph_pre | 13.14 | 13.14 | 13.18 | 1772 | 1 | **13.14** |
| target | graph_post | 5.98 | 5.98 | 6.00 | 1772 | 1 | **5.98** |
| target | proxy_compute_send | 1.79 | 1.70 | 1.98 | 1772 | 1 | **1.79** |
| draft | glue | 4.94 | 4.94 | 4.99 | 1772 | 1 | 4.94 |
| draft | phase1_replay | 4.17 | 4.17 | 4.23 | 7103 | 4 | **16.68** |
| draft | phase2_replay | 4.20 | 4.21 | 4.24 | 7103 | 4 | **16.80** |
| draft | proxy_wait | 0.003 | 0.003 | 0.003 | 1772 | 1 | 0.003 |
| draft | merge_cache | 0.065 | 0.065 | 0.067 | 1772 | 1 | 0.065 |

## Per-spec-step totals

- **Target**: `graph_pre + proxy + graph_post` = **20.9 ms/step** (vs baseline ~23 ms verify — split CudaGraph self-cost negligible)
- **Draft (measured)**: `glue + 4×phase1 + 4×phase2 + merge + wait` = **38.5 ms/step**
- **Draft (observed in metrics)**: 43.4 ms — delta ~5 ms from unmeasured mask+buf / plan / python overhead
- **Baseline draft (observed)**: 24 ms — delta = 38.5 - ~22 ≈ **~16-17 ms** which is precisely one full Phase-2 replay cycle (4 × 4.2)

## Key findings

1. **`proxy_wait` ≈ 3 µs** — target's proxy irecv handshake arrives **before** draft needs it. Phase-1 fully hides the wait. The pipeline works.
2. **Structural overhead of MESA = Phase-2 replay cycle (16.8 ms)**. This is the single dominant source of the 35-40 % slowdown vs baseline SSD.
3. **Target split-CudaGraph has no measurable overhead**: `graph_pre` (22 layers, 13.1 ms) + `graph_post` (10 layers, 6.0 ms) + proxy (1.8 ms) = 20.9 ms — *less* than baseline full-graph verify. The split itself is free; it's the draft's 2-pass cost that dominates.
4. **`graph_pre` dominates target (63 %)**, matches 22/32 layer ratio. No surprises from CudaGraph split.

## Interpretation

- On this 8B / RTX 3090 config, MESA's token-efficiency gain (+0.03-0.05 cache hit) is not enough to offset the ~17 ms/step Phase-2 tax.
- Directions that would flip the verdict, in rough order of expected impact:
  1. Fuse Phase-1 and Phase-2 into a **single** proxy-aware tree decode (eliminate Phase-2 entirely, ~17 ms saving).
  2. Make Phase-2 much smaller (only a few "enrichment" branches, not a full K×F redo).
  3. Scale up target (70B): `graph_pre` and `graph_post` become much more expensive, Phase-2 on 1B draft stays cheap — ratio flips.
