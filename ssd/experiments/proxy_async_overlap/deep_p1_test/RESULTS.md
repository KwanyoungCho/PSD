# Deep-P1 test — token model perfect, time tax 2.7× projection → rejected as-is

**Date**: 2026-07-02
**Setup**: same as best_config_rematch. K2=5 exit=56 dfo=2 pfo=1 fixed;
K1 ∈ {7 (baseline), 8, 9, 10}. 1 rep per cell. Zero code changes
(jit_K follows K_max — a real implementation would pin it at 7, worth
~+0.4-0.7 ms/step here, not enough to change the verdict).

## Headline

| K1 | k | TPS | tok/step | L_p1 meas (geometric model) | target step | draft step |
|---:|---:|---:|---:|---|---:|---:|
| 7 | 12 | **81.24±0.67** | 4.02 | 3.67 (fit point) | 51.15 | 46.01 |
| 8 | 13 | 78.48 | 4.16 | **3.92 (3.91)** | 54.38 | 50.63 |
| 9 | 14 | 75.59 | 4.27 | **4.12 (4.12)** | 57.91 | 54.35 |
| 10 | 15 | 72.55 | **4.38** | **4.35 (4.29)** | 61.71 | 58.15 |

## Finding 1 — the token thesis is CONFIRMED, three cells in a row

The truncated-geometric acceptance model (α = 0.838 fitted from the
K1=7 point alone) predicted L_p1 to within 0.01-0.06 at every depth.
P1/P2 hit shares stay flat (no substitution — deepening raises P1 hit
QUALITY, unlike widening which cannibalized P2). tok/step at K1=9-10
(4.27-4.38) **exceeds SD-best C (4.15)** — DUET can out-produce the
baseline in tokens.

## Finding 2 — the per-position time tax is 3.3-3.4 ms, not the
projected 1.26 ms

target step grows +3.3-3.4 ms per K1 (linear, three-cell fit), against
a projected +1.26 (verify position slope measured from the hit_k1 vs
hit_k2 CG buckets) + ~0.35 (deeper JIT on miss). Roughly ~1.7-2 ms per
position is unexplained by verify alone. Draft step grows +4.0 per K1
(phase1 forward +2.5, remainder unattributed — likely response wire
[B,K,V] logits_q growth, prep, and pipeline echo).

TPS check: (4.27/4.02) × (51.15/57.91) = 0.938 → 76.2 predicted vs
75.6 measured ✓ — the decomposition is internally consistent.

## Verdict

Deep-P1 is **rejected under the current cost structure**, but for the
opposite reason to widening: widening failed on tokens (substitution),
deepening succeeds on tokens and fails on time. The break-even table
for the per-position tax:

| per-pos tax | TPS @ K1=9 | vs C (82.72) |
|---:|---:|---|
| 3.4 ms (now) | 75.6 | loses |
| 1.7 ms | ~78.2 | loses |
| 1.26 ms (verify only) | ~79.5 | loses |
| ~0.6 ms | ~82.7 | ties |
| 0 | 83.5 | wins |

So deep-P1 becomes viable only if the pipeline's per-position cost is
cut ~5×. That is a systems investigation (where does 3.4 ms/pos go —
verify CG bucket scaling on Marlin small-batch, logits_q NCCL,
verify_sample_accept K-linear work, proxy compute K-scaling), not an
algorithm change.

## Where this leaves the algorithm question

1. Widening (dfo/pfo): rejected — substitution effect.
2. Deepening (K1): token-correct, time-rejected at 3.4 ms/pos.
3. DUET (2,1) K1=7 K2=5 is the local optimum of the current design:
   81.24 vs C 82.72 (−1.8%).
4. The remaining levers are all systems-side: (a) decompose and shrink
   the 3.4 ms/pos tax (PROFILE run at K1=9 would attribute it), (b) pin
   JIT depth at 5-7 (+1%), (c) glue-removal/KV-promotion only as an
   enabler if (a) succeeds and Window 1 becomes binding.
5. Algorithm-side, the fundamental remaining gap vs C is L_p2 ≈ 2.0
   off-policy continuation (breakeven 2.6) — draft adaptation territory.
