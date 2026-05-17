# MESA K1=K2=7 exit=52 — dfo × pfo sweep (PROFILE_MESA=0)

**Run dates**: 2026-05-17 16:11 → 18:58 (2h 47m wall — much shorter than
the projected 5-6h because 9/12 cells crashed early).
**Branch**: feat/mesa-proxy-async-overlap @ 8874009
**Config**: 70B AWQ TP=4 target + TinyLlama-1.1B AWQ TP=1 draft, ns=50
            in=512 out=512, seed=42, temp=0.7, `--async --spec --k 14
            --mesa --mesa_phase1_k 7 --mesa_phase2_k 7
            --mesa_exit_layer 52 --mesa_policy b`, SSD_FORCE_SPLIT_K1K2=1,
            SSD_PROFILE_MESA=0.
            Per cell: `--f $(dfo + pfo)`, `--mesa_draft_fan_out $dfo`.

## Headline — decode_tps (tok/s)

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| **dfo=2** | **77.94** | 71.14 |  —    |
| **dfo=3** | 75.38 |  —    |  —    |
| **dfo=4** |  —    |  —    |  —    |
| **dfo=5** |  —    |  —    |  —    |

— = run crashed before producing decode_tps.

**Successful cells (3/12)**:
| (dfo, pfo) | f | TPS | step_ms | cache_hit | accept |
|---|---:|---:|---:|---:|---:|
| (2, 1) | 3 | 77.94 | 54.52 | 0.80 | 0.45 |
| (2, 2) | 4 | 71.14 | 59.14 | 0.82 | 0.44 |
| (3, 1) | 4 | 75.38 | 54.81 | 0.82 | 0.43 |

## 🔴 Bug — every cell with `f = dfo + pfo ≥ 5` crashes

9/12 cells fail with the SAME stack trace:

```
target rank 0:
  speculator_async.py:155
    vk = int(valid_k[0].item())
  IndexError: index 0 is out of bounds for dimension 0 with size 0

draft worker (cascading):
  cudagraph_helpers.py:53  run_verify_cudagraph
    block_tables = torch.cat([bt, bt[orig_bs-1:orig_bs].expand(pad_bs, -1).contiguous()])
  RuntimeError: The expanded size of the tensor (1) must match the existing size (0)
                 at non-singleton dimension 0.  Target sizes: [1, -1].  Tensor sizes: [0, 8]
```

Crashed cells (all f ≥ 5):

| (dfo, pfo) | f | wall time before crash |
|---|---:|---:|
| (2, 3) | 5 | 27 min |
| (3, 2) | 5 | 25 min |
| (3, 3) | 6 |  5 min |
| (4, 1) | 5 | 22 min |
| (4, 2) | 6 |  5 min |
| (4, 3) | 7 |  2 min |
| (5, 1) | 6 |  5 min |
| (5, 2) | 7 |  2 min |
| (5, 3) | 8 |  2 min |

The crash latency varies because some cells get further into CG capture / warmup
before the empty-`valid_k` condition fires. All crashes are in the MESA decode
phase (`_build_tree_batch_split_k1k2` → `_glue_decode` → `_run_split_k1k2_glue`).

The empty `block_tables [0, 8]` and empty `valid_k` together suggest the spec
request response is arriving with `B=0` (no rows) at the speculator, which
cascades into the draft glue trying to dispatch a CG for an empty batch.

**Hypothesis**: at higher `f` (= async_fan_out), the per-row scheduler decides
to defer all sequences to a later step (e.g. block budget exhaustion), so the
spec request reaches the speculator with `B=0`. The split-K1K2 path doesn't
guard against this — every path assumes `B≥1`. This is independent of
PROFILE_MESA setting (PROFILE_MESA=0 in this sweep; same crash would occur at
PROFILE_MESA=1).

This affects MESA split-K1K2 mode specifically; the async SD sweep
(`../async_sd_sweep/`) at f=5/6 did NOT crash because the scheduler's
empty-B path is handled by the standard verify CG, not the split-K1K2 glue.

## Limited findings from the 3 successful cells

1. **Best**: dfo=2 pfo=1 (f=3) → **77.94 tok/s**.
2. **Increasing dfo at fixed pfo=1**: 77.94 (dfo=2) → 75.38 (dfo=3)
   → dfo↑ hurts TPS slightly. Step time ~unchanged, but accept_fraction
   dropped 0.45 → 0.43 (more draft branches dilute per-pos accuracy).
3. **Increasing pfo at fixed dfo=2**: 77.94 (pfo=1) → 71.14 (pfo=2)
   → pfo↑ hurts more dramatically. step_time 54.52 → 59.14 ms (+4.6).
   Cache hit barely budges (0.80 → 0.82), so the extra proxy branches
   are paying verify cost without proportional gain.
4. **Cache hit rate ~0.80-0.82**: much higher than async SD f=3 (0.66 at
   same f=3), but TPS is LOWER (77.94 vs 80.35). MESA's tree-fill quality
   is real but not enough to overcome the K=14 verify cost on this draft.

## Comparison vs prior reference points

| run | mode | K | f / fan-out | exit | TPS |
|---|---|---|---|---|---:|
| async_sd_sweep best | SD only | k=7 | f=6 | n/a | **83.65** |
| async_sd_sweep | SD only | k=7 | f=3 | n/a | 80.35 |
| k2_5_tps_verify B | MESA | K1=7 K2=5 (k=12) | dfo=2 pfo=1 (f=3) | 56 | 80.42 |
| this sweep best (3/12 valid) | MESA | K1=K2=7 (k=14) | dfo=2 pfo=1 (f=3) | 52 | 77.94 |

At fixed (dfo, pfo) = (2, 1), exit=52 K1=K2=7 is **2.48 tok/s slower than
exit=56 K1=7 K2=5** (77.94 vs 80.42). Two changes between these:
- K2: 5 → 7 (deeper proxy tree, larger Phase 2 forward cost)
- exit_layer: 56 → 52 (earlier target exit, faster target verify)

The early-exit gain (target step 50.76 → 54.52 ms is actually +3.76 ms
SLOWER, contradicting expectations — likely because the larger K=14
verify CG outweighs the saved exit-layer layers) does NOT cover the
larger K2 cost.

## Followups (not run in this commit)

1. **Fix the f≥5 split-K1K2 crash.** The empty-B condition needs to be
   guarded in `_glue_decode` / `_build_tree_batch_split_k1k2`. Once fixed,
   re-run the full 12-cell sweep.

2. **Compare exit_layer**: holding K1=K2=7, sweep exit_layer ∈ {48, 52, 56,
   60} at the best (dfo, pfo) to find the sweet spot for split-K1K2 with K=14.

3. **Comparison fairness**: the existing async SD sweep used k=7 f=6.  A
   fair MESA comparison would tune MESA to similar total k. Possible
   configurations: K1=4 K2=3 (k=7), K1=5 K2=2 (k=7).

## Artifacts

```
mesa_k1k2_7_exit52_dfo_pfo_sweep/
  run_sweep.sh                # driver
  analyze_sweep.py            # 4×3 grid extractor
  sweep_master.log            # phase markers
  dfo{2..5}_pfo{1..3}/run.log # per-cell bench output (incl crash tracebacks)
  dfo*_pfo*/headline.txt      # per-cell extracted metrics (mostly partial for crashed cells)
  sweep_grid.csv              # long-form, all metrics
  sweep_all.md                # all grids
  RESULTS.md                  # this file
```
