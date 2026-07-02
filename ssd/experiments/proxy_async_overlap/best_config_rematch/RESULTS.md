# Best-config rematch — A (DUET) vs C/D (async SD), 3 reps each

**Date**: 2026-07-02
**Branch**: feat/mesa-proxy-async-overlap @ f619a67 (post Batch 1/2/3)
**Setup**: 70B AWQ TP=4 (GPU 0-3) + TinyLlama AWQ TP=1 (GPU 4), ns=50
in=512 out=512 --all, seed=42 temp=0.7, SSD_PROFILE_DUET=0 (cold path).
Configs interleaved across reps (A→C→D per cycle) to spread drift.

## Headline (decode TPS)

| config | rep1 | rep2 | rep3 | mean | std | CoV | prior | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A DUET K1=7 K2=5 exit=56 dfo=2 pfo=1 (f=3) | 80.54 | 81.31 | 81.88 | **81.24** | 0.67 | 0.8% | 80.42 | +1.0% |
| C SD k=7 f=6 | 83.19 | 82.53 | 82.44 | **82.72** | 0.41 | 0.5% | 83.65 | −1.1% |
| D SD k=7 f=3 | 78.72 | 80.18 | 82.05 | **80.32** | 1.67 | 2.1% | 80.35 | −0.0% |

## Statistical verdicts (pre-registered rule: bands must not overlap)

- **C vs A: C wins, significant.** [82.31, 83.13] vs [80.57, 81.92] — no
  overlap. Gap +1.48 tok/s (+1.8%).
- **A vs D: statistical tie.** D's band [78.65, 81.99] contains A's mean.
  D's run-to-run variance is large (3.3 tok/s span across reps) — the
  f=3 SD config is noisier than both A and C. DUET does not lose at
  f-matched, but the +0.9 advantage is inside noise.
- **Code-change check**: all three configs within ±1.1% of their May
  measurements → Batch 1 (sync removal) + Batch 2 (−3.8k LOC) caused
  no regression; A is +1.0% (direction consistent with the sync fixes).

## Per-status supporting data (rep1 metrics + k2_5_tps_verify profile)

A-config step composition (from the PROFILE=1 K1=7 K2=5 profile,
25,571 steps):

| status | share | target T_total | draft proxy_wait | notes |
|---|---:|---:|---:|---|
| hit_k1 | 0.596 | 48.90 ms | 8.90 ms | verify 8 pos |
| hit_k2 | 0.219 | 44.15 ms | 5.47 ms | verify 6 pos — target ends earlier AND proxy arrives earlier; spec_wait identical (2.36 vs 2.42) |
| miss | 0.184 | 60.77 ms | 8.77 ms | +13 ms JIT in spec_wait |

Draft work is constant across statuses (7.0 + 5.0 forwards per step).
Draft windows (hit_k1): glue+phase1 = 22.4 ms vs proxy arrival ~29 ms
(6.6 ms slack); phase2+merge = 12.5 ms vs post-proxy budget ~17.5 ms
(5.0 ms slack).

## Interpretation

1. DUET's value is confirmed as **miss-avoidance latency** (L_p2 1.99 ≈
   L_miss 2.18; the population token yield matches f-matched SD as
   expected mathematically). At f=3 DUET ≥ SD.
2. The remaining deficit vs SD's best operating point (f=6) is **pure
   tree width** — target latency is equal (51.15 vs 51.20 ms mean).
   SD gets f=6 for free because its draft has 18 ms slack; DUET's draft
   has ~5-6.6 ms slack in each of its two windows.
3. Next: window-budget-guided DUET widening — (3,1) f=4, (4,1) f=5,
   (4,2) f=6, then K2=3 variant of the best. These cells were
   unreachable before the B=0 guard + lookahead fixes.
