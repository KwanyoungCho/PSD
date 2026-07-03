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

E9K24_jit vs C, 5 cycles, alternating. Rule: DUET mean > C mean AND
mean±2SE bands must not overlap. (results appended on completion)
