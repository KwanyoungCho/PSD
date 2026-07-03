# Final rematch — probe phase: three nulls expose the binding side; K2=4 breaks the tie

**Date**: 2026-07-03
**Setup**: canonical GPU set 0-4 (target TP4 on 0-3, draft on 4), ns=50
in=512 out=512 --all, seed=42 temp=0.7, PROFILE=0. A neighbor vLLM
occupied GPUs 6-7 from ~13:30 (and 0-1 before ~11:20); interleaving and
same-regime pairing are used for every claim. `altset_partial/` holds
the aborted 2,3,5,6,7-set reps.

## Phase 1 — 3-rep E9_jit vs C: exact tie

| arm | reps | mean | tok/step | T_target |
|---|---|---:|---:|---:|
| E9_jit | 83.34 / 81.35 / 81.91 | 82.20 ± 1.03 | 4.213 | 52.59 |
| C (k7 f6) | 82.44 / 84.37 / 79.66 | 82.16 ± 2.36 | 4.130 | 51.56 |

Token premium +2.01% vs time premium +2.00% — structural difference
**0.0%**. T_target is rock-stable within arm (±0.13); ALL the TPS spread
is token-side sampling luck at temp=0.7 (C_rep3 also ate the neighbor's
return). Verdict: not decidable, and E9_jit as-is does not beat C.

## Phase 2 — three attempts to shave the DUET-only target cost: ALL NULL

| probe | idea | TPS | T_target | verdict |
|---|---|---:|---:|---|
| E9jit_pod (old) | Policy B → draft idle | 80.07 | 53.82 | +1.2ms REGRESSION (reproduced 3×) |
| E9jit_pod2 (gather-based p_D) | shorten the local_b chain | 81.23* | 54.64 | regression persists |
| E9jit_front | width→pos0 `[3,2,2,2,2,1×5]` | 80.73 | 52.72 | tokens LOST (L_p1 4.02) |
| E9jit_topm | rank-local top-M exit gather (640KB→16KB) | 81.14 | 54.10 | time unchanged |
| E9jit_replica | rank-0 lm_head replica + side stream — NO collective at all | 80.99 | 53.55 | −0.14ms only |

(* first pod2 attempt crashed: stale `p_D.device` NameError, fixed 793243d.)

The consistency is the finding: **at E9 the target-busy chain no longer
binds.** topm cut the exit gather volume 40× → nothing; replica removed
the exit collective entirely → nothing. The mid-verify DUET overhead
sits in slack. The May "exit_logits = rendezvous wait" observation
generalizes: what looks like target-side cost is the coupled pipeline
breathing. Deepening to K1=9 grew draft busy to ~47.5ms ≈ co-critical
with target busy (~50ms) — savings on either side alone vanish into
the other side's wait. (Corollary: proxy quality was preserved in every
probe — p2_hit 0.24-0.25, L_p2 1.99-2.09 — the candidate-set Policy B
math is production-correct.)

## Phase 3 — cut the DRAFT chain instead: K2 shrink under jit-short economics

jit-short changed K2's price list: K2 now also sets the miss JIT depth
AND the miss/short verify width. Same-regime (neighbor present) numbers:

| cell | TPS | tok/step | L_p2 | T_target | T_draft |
|---|---:|---:|---:|---:|---:|
| E9_jit (rep4, baseline) | 80.09 | 4.19 | ~2.0 | 53.69 | 48.6 |
| **E9K24_jit (K2=4)** | **83.34** | 4.17 | 1.76 | **51.34** | **45.28** |
| E9K23_jit (K2=3) | 82.05 | 4.03 | 1.56 | 50.42 | 41.33 |

K2=4: draft −4.4ms propagates to period −2.35ms at a token cost of only
−0.04 tok/step → **+3.3 tok/s same-regime**. K2=3 overshoots (P2 tokens
fall faster than time). This confirms the co-critical structure: the
draft-side cut moved the period where three target-side cuts could not.

## Champion

**E9K24_jit**: split-K1/K2, K1=9, K2=4 (k=13), exit=56, pfo=1, dfo=2,
phase1 fan_out_list `2,2,2,2,2,2,1,1,1,1` (sum 16 = draft Marlin-tile
budget), `SSD_DUET_JIT_SHORT=1`. Same-regime margin over C ≈ +3.7.

## Phase 4 — VERDICT (5-rep interleaved, pre-registered)

| cycle | E9K24_jit | C | paired diff |
|---|---:|---:|---:|
| 1 | 83.93 | 81.54 | +2.39 |
| 2 | 83.11 | 82.52 | +0.59 |
| 3 | 81.51 | 80.18 | +1.33 |
| 4 | 80.31 | 80.10 | +0.21 |
| 5 | 80.70 | 83.24 | −2.54 |
| **mean** | **81.91 ± 1.56** | **81.52 ± 1.39** | **+0.40** (4/5 wins, paired t=0.48) |

**Pre-registered rule NOT met**: DUET mean > C mean ✓ but ±2SE bands
overlap heavily. Structural decomposition: tok 4.108 vs 4.098 (+0.24%),
T_target 51.44 vs 51.57 (−0.26%) → true edge ≈ **+0.5%**, which at
σ_pair 1.84 would need ~54 cycles to certify — not provable by reps.

## Phase 5 — post-verdict probes: the frontier is real

| probe | idea | TPS | tok | T_target | verdict |
|---|---|---:|---:|---:|---|
| E9K24R | champion + exit-replica (target-bound again?) | 78.99 | 3.98 | 51.61 | null AGAIN — the mid-verify block is never load-bearing |
| E10K24_jit | K1=10 deep-narrow `[2×5,1×6]` | 81.76 | 4.25 | 53.32 | tok +3.7% eaten by time +3.4% — frontier slide |
| E9K24P2 | pfo=2 (P2 coverage 2×, no verify-width cost) | 78.14 | 4.04 | 52.95 | p2_hit +2.6pp but p1_hit −1.7pp (substitution) + draft co-critical — net −0.5% |

## Campaign conclusion

**Final config: E9K24_jit** — split-K1/K2, K1=9 K2=4 (k=13), exit=56,
dfo=2 pfo=1, phase1 fan_out_list `2,2,2,2,2,2,1,1,1,1` (sum 16),
SSD_DUET_JIT_SHORT=1.

- DUET itself improved **+6.0%** today (A_base 77.30 → 81.91 champion
  mean): jit-short +3.85, deep-narrow depth tokens, K2=4 sweet spot.
- vs SD-best C: from **−1.8% (morning) to +0.5% (evening)**, winning
  4/5 interleaved cycles — but not band-clear at feasible rep counts.
- The deep reason (user's standing question) is now measured, not
  conjectured: on this hardware the coupled pipeline charges ~1.9 ms
  per verify position while DUET's marginal position yields ~1.9 ms of
  tokens at viable depths — the design frontier passes through C.
  DUET's real, certified advantages are (a) the miss-latency tail
  (miss share 0.15 vs 0.24, each miss ~9.5 ms JIT vs ~13 ms) and
  (b) reaching SD-best throughput at f=3 vs f=6 — the
  draft-compute-constrained regimes are DUET's ground.
- Five independent null probes (pod ×2, topm, replica ×2) plus two
  frontier probes (E10, pfo=2) bound the systems-side search space;
  the remaining lever is algorithmic: off-policy continuation quality
  (L_p2 ≈ 2.0 vs breakeven 2.6) — draft adaptation territory.
