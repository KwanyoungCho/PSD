# MESA proxy_compute_send async-overlap optimization

Working directory for the target-side proxy send overlap work.
See conversation notes / `docs/mesa/` for design.

## Phase structure

- `phase0b/` — A/B measurement (DETAIL=0 vs DETAIL=1). No engine changes.
- `phase1/`  — TPS-oriented metric + plot color separation (analysis only).
- `phase2/`  — Async proxy send + ring buffer (env-gated, default OFF).
- `phase3/`  — proxy_stream overlap (env-gated, default OFF).
- `phase4/`  — Default-on decision artifacts.

Env gates:
- `SSD_ASYNC_PROXY_SEND=1` — enable async ring-buffered isend (Phase 2)
- `SSD_PROXY_STREAM=1`      — enable proxy_stream overlap (Phase 3)

Both default OFF until validated.

## Reproduction commands

Baseline K1=K2=7 metadata (verbatim source):
  `experiments/paper_baselines/final_experiments/20260513_ours_k1_7_k2_7_dfo2_pfo1_exit56_temp07_seed42_ns50_in512_out512_all/metadata.txt`

Phase 0b A/B (sequential, ~30 min total on RTX 3090 ×5):
```
bash experiments/proxy_async_overlap/phase0b/run_ab.sh
python experiments/proxy_async_overlap/phase0b/analyze.py
```

STOP conditions (any one → halt and report):
1. `|A.outer_mean − B.outer_mean| > 0.5 ms` (DETAIL probe perturbs outer)
2. `unattributed_stall < 0.5 ms` (no headroom — wrong bottleneck)
3. Phase 2/3 results: `proxy_outer ↓` but `target_spec_wait ↑` matching (wait shift)
4. Phase 2/3 results: greedy byte-identical fails any seed
