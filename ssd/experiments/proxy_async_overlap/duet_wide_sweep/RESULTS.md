# DUET wide sweep — widening hypothesis REJECTED, substitution effect measured

**Date**: 2026-07-02
**Setup**: same as best_config_rematch (70B AWQ TP=4 + TinyLlama, ns=50
in=512 out=512, seed=42 temp=0.7, PROFILE_DUET=0). 1 rep per cell (scan).
K1=7 K2=5 exit=56 unless noted.

## Headline

| cell | f | TPS | tok/step | cache | p1_hit / L_p1 | p2_hit / L_p2 | miss | T_target | T_draft |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| **A (2,1) [3-rep baseline]** | 3 | **81.24±0.67** | 4.02 | 0.81 | 0.592 / 3.67 | 0.220 / 1.99 | 0.188 | 51.15 | 46.01 |
| (3,1) | 4 | 79.58 | 4.04 | 0.84 | 0.657 / 3.55 | 0.181 / 1.96 | 0.162 | 52.11 | 48.35 |
| (4,1) | 5 | 80.89 | 4.06 | 0.86 | 0.703 / 3.50 | 0.158 / 1.85 | 0.139 | 51.48 | 46.47 |
| (4,2) | 6 | 80.41 | 4.08 | 0.88 | 0.704 / 3.53 | 0.172 / 1.90 | 0.124 | 52.02 | 48.33 |
| (4,1) K2=3 (k=10) | 5 | 80.41 | 3.96 | 0.86 | 0.701 / 3.49 | 0.159 / **1.42** | 0.140 | **50.48** | **41.03** |
| C SD k=7 f=6 [3-rep] | 6 | **82.72±0.41** | 4.15 | 0.76 | — | — | 0.24 | 51.20 | 38.57 |

All four new cells ran to completion — the B=0 guard + lookahead fixes
hold at f=4-6 (these all crashed in May).

## Finding 1 — widening DUET is flat (substitution effect)

dfo 2→4 raises p1_hit +0.111 (0.592→0.703) but p2_hit falls −0.062
(0.220→0.158): **more than half of the new P1 coverage is cannibalized
from P2, not from misses.** Miss only drops −0.049. The proxy was
already covering the easy part of the miss pool; widening the draft fork
re-covers the same tokens through a different door. Net tok/step gain:
+0.04 (4.02→4.06) — wiped out by +0.3-1.0 ms step time. TPS flat to
slightly negative across all widened cells.

## Finding 2 — the marginal-hit quality ladder

Marginal L of each miss→hit conversion mechanism (derived from D→C and
A's phase split):

| conversion mechanism | marginal L (excl. recovery) |
|---|---:|
| draft rank 1-3 fork (D's hits) | ≈ 3.6 |
| draft rank 4-6 fork (C's marginal hits over D) | ≈ 3.0 ← (0.76×3.49 − 0.66×3.56)/0.10 |
| proxy residual seed (DUET P2) | ≈ 2.0 |
| (stay a miss, JIT chain) | ≈ 2.1 + 13 ms stall |

Both SD-widening and DUET-P2 harvest the same miss pool. SD's harvest
is on-policy (draft continues from a token it ranked 4-6 → L≈3.0);
P2's is off-policy (draft continues from a token it ranked ~7+ → L≈2.0,
essentially a cached JIT chain minus the 13 ms). P2 conversions still
beat misses on rate (67.7 vs 52.3 tok/s per step) — that is why A ≥ D —
but lose to SD width-conversions on tokens — that is why C > A.

## Finding 3 — K2=3 is a wash

L_p2 truncates 1.99→1.42 (−0.10 tok/step) but frees 5 ms of draft time
(41.03 vs 46.47) and 1 ms of target step. TPS 80.41 vs 80.89 — neutral.
The freed slack has no consumer; would only matter combined with a
feature that uses it.

## Verdict on "is my method better"

- **vs f-matched SD (D)**: DUET ties-to-wins (81.24±0.67 vs 80.32±1.67)
  with fewer JIT stalls (miss 0.19 vs 0.34) and visibly lower run
  variance (CoV 0.8% vs 2.1%).
- **vs SD's best point (C, f=6)**: DUET loses by 1.48 tok/s (1.8%),
  significant. Widening DUET does not close it (Finding 1).
- The residual gap is **fundamental to off-policy continuation quality**:
  breakeven needs L_p2 ≈ 2.6 (currently 2.0), i.e. the draft would have
  to continue ~30% better from tokens it considers unlikely.

## Remaining levers (not yet tried)

1. **exit_layer sweep** (56 → 60-64): later exit = better proxy = higher
   p2_hit. Each +1pp p2_hit (from miss) ≈ +0.02 tok/step equivalent +
   12 ms avoided. Does not fix L_p2 but grows P2's reach.
2. **Latency-jitter framing**: DUET's miss rate (0.19 vs 0.24-0.34)
   directly reduces the 60 ms step tail — p99 token latency should
   favor DUET even where mean TPS ties. Measurable from existing
   PROFILE data.
3. **Regimes where SD cannot afford f=6** (bigger draft model, shared
   draft GPU, B>1): DUET reaches 0.81/0.86 hit at f=3-5; SD needs f=6
   for 0.76. Draft-compute-constrained setups are DUET's natural ground.
4. Draft adaptation (train/LoRA the draft on off-policy continuations) —
   out of scope for a systems paper.
