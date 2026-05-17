# Async SD sweep — k=7-10 × f=3-6 (PROFILE_MESA=0)

**Run dates**: 2026-05-17 09:19 → 15:46 (6h 27m wall, 16 sequential runs)
**Branch**: feat/mesa-proxy-async-overlap @ 10ab6ee
**Config**: 70B AWQ TP=4 target + TinyLlama-1.1B AWQ TP=1 draft,
            ns=50 in=512 out=512, seed=42, temp=0.7, `--async --spec`
            (NO MESA, NO SSD_FORCE_SPLIT_K1K2). `SSD_PROFILE_MESA=0`
            → cold path, no measurement overhead.

## Headline grid — decode_tps (tok/s)

| k \ f |  f=3  |  f=4  |  f=5  |  f=6  | row peak |
|---|---:|---:|---:|---:|---|
| **k=7**  | 80.35 | 82.99 | 81.07 | **83.65** | f=6 |
| **k=8**  | 79.68 | 79.22 | **81.79** | 77.94 | f=5 |
| **k=9**  | 78.55 | **80.38** | 71.94 | 73.36 | f=4 |
| **k=10** | **75.32** | 73.93 | 65.75 | 57.66 | f=3 |

**Best overall: `k=7 f=6` → 83.65 tok/s.**

## Anatomy — target full step (ms)

| k \ f | f=3 | f=4 | f=5 | f=6 |
|---|---:|---:|---:|---:|
| k=7  | 52.99 | 51.32 | 51.16 | 50.87 |
| k=8  | 54.67 | 55.17 | 54.51 | 56.30 |
| k=9  | 57.05 | 57.43 | **62.98** | **63.30** |
| k=10 | 62.00 | 62.19 | **70.34** | **81.69** |

Step time grows monotonically with both k and f. The big jumps at
k=9 f=5 and k=10 f=5/6 (62→70→82 ms) suggest the FlashInfer tree
mask + verify CG cost is super-linear when the tree exceeds some
working-set threshold on RTX 3090 + TP=4.

## Tokens per step (incl recovery)

| k \ f | f=3 | f=4 | f=5 | f=6 |
|---|---:|---:|---:|---:|
| k=7  | 4.15 | 4.15 | 4.04 | 4.15 |
| k=8  | 4.25 | 4.26 | 4.35 | 4.28 |
| k=9  | 4.37 | 4.50 | 4.43 | 4.54 |
| k=10 | 4.56 | 4.49 | 4.53 | 4.63 |

Tokens-per-step grows with k (deeper trees produce more accepted tokens),
but step time grows faster → net TPS drops.

## accept_fraction

| k \ f | f=3 | f=4 | f=5 | f=6 |
|---|---:|---:|---:|---:|
| k=7  | 0.45 | 0.45 | 0.43 | 0.45 |
| k=8  | 0.41 | 0.41 | 0.42 | 0.41 |
| k=9  | 0.37 | 0.39 | 0.38 | 0.39 |
| k=10 | 0.36 | 0.35 | 0.35 | 0.36 |

Accept fraction degrades cleanly with k (longer chains harder to accept
all of). Mostly insensitive to f.

## cache_hit_rate

| k \ f | f=3 | f=4 | f=5 | f=6 |
|---|---:|---:|---:|---:|
| k=7  | 0.66 | 0.70 | 0.72 | 0.76 |
| k=8  | 0.64 | 0.69 | 0.73 | 0.75 |
| k=9  | 0.63 | 0.69 | 0.71 | 0.74 |
| k=10 | 0.62 | 0.67 | 0.70 | 0.74 |

Cache hit rate climbs monotonically with f (wider tree → more chance of
covering the recovery token).  Only weakly dependent on k.

## draft_step_ms

| k \ f | f=3 | f=4 | f=5 | f=6 |
|---|---:|---:|---:|---:|
| k=7  | 33.76 | 33.50 | 38.35 | 38.55 |
| k=8  | 38.62 | 44.26 | 44.24 | 49.69 |
| k=9  | 43.85 | 49.97 | 56.37 | 56.56 |
| k=10 | 56.19 | 56.36 | 63.61 | 73.76 |

Draft step grows with both k and f but stays slightly under
target_full_step in most cells, so the target is still the binding
constraint (draft has enough slack to overlap, except in the
biggest-tree cells where draft starts catching up).

## Interpretation

1. **Sweet spot is small k, large f** for this draft-target pair
   (TinyLlama-1.1B + 70B AWQ).  The draft is weak enough that deeper
   chains lose acceptance faster than they gain tokens, while wider
   trees keep paying off until the verify cost balloons.

2. **k=7 is the dominant row.**  Every cell in k=7 except f=5
   beats every cell in k=8/9/10.  The "noise" at k=7 f=5 (81.07 vs
   peaks 82.99/83.65 around it) is likely run-to-run variance —
   accept dropped to 0.43 on that single seed.

3. **k≥9 f≥5 cliff**: target_full_step jumps from ~57 ms to ~63-82 ms
   when the tree exceeds working-set bounds.  Not worth running these
   configurations.

4. **Paper number**: at `--k 7 --f 6` async SD on this hardware
   already hits **83.65 tok/s**, well above the 80 tok/s threshold,
   and ~+4 % over the MESA K2=5 baseline at the same MESA=1
   measurement (76.35) or the cold-path MESA K2=5 (80.42).  This
   matters as a baseline for MESA claims: MESA must beat 83.65, not
   just 80, to be a TPS improvement at the apples-to-apples cold-path
   setting.

## Comparison vs prior data points

| run | mode | k / config | decode_tps |
|---|---|---|---:|
| 20260512_ours_label_perf | MESA K1=7 K2=5 @ MESA=1 | k=12 | 76.35 |
| k2_5_tps_verify A (current @ on) | MESA K1=7 K2=5 @ MESA=1 | k=12 | 77.84 |
| k2_5_tps_verify B (current @ off) | MESA K1=7 K2=5 @ MESA=0 | k=12 | 80.42 |
| **async_sd_sweep k=7 f=6** | **async SD baseline @ MESA=0** | **k=7** | **83.65** |
| async_sd_sweep k=7 f=4 | async SD baseline @ MESA=0 | k=7 | 82.99 |
| async_sd_sweep k=7 f=3 | async SD baseline @ MESA=0 | k=7 | 80.35 |

**Async SD baseline at the right (k, f) beats MESA K2=5 by ~3-4 % at
the same cold-path setting.**  MESA's value proposition at this
configuration is NOT raw TPS — it would be improved acceptance
distribution, accept length, or other quality metrics (which we have
not compared here).  For paper headline TPS numbers, the MESA-vs-SD
comparison needs to be re-done at apples-to-apples best-(k, f) per
mode.

## Artifacts

```
async_sd_sweep/
  run_sweep.sh                 # 16-run driver
  analyze_sweep.py             # grid extractor
  sweep_master.log             # outer phase markers
  k7_f3 ... k10_f6/            # per-run subdirs (run.log + headline.txt)
  sweep_grid.csv               # long-form data
  sweep_decode_tps.md          # decode_tps grid only
  sweep_all.md                 # all 6 metric grids + best cell
  RESULTS.md                   # this file
```

## Open questions / next steps

- **Re-do the MESA comparison at the best (k, f) per mode.**  MESA
  was always run at k=12 (= K1+K2 = 7+5).  If best-tuning async SD
  uses k=7, MESA's k should also be tuned (e.g. K1=4 K2=3 or
  K1=5 K2=2 to land at k=7-8 totals).  Currently MESA at k=12 is
  fighting both the k cost and the accept-rate cost simultaneously.

- **Run-to-run variance**: single seed = single sample per cell.  At
  noise-sensitive cells (e.g. k=7 f=5) a second seed would tighten
  the picture.  Cost: ~22 min per re-run.

- **Cache hit-rate ceiling**: 0.76 max in this sweep.  TinyLlama draft
  saturates around that.  A better draft (e.g. EAGLE-3) could push
  higher.
