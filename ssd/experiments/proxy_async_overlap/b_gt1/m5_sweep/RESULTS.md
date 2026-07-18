# M5 — B ∈ {1,2,4} sweep: DUET champion vs SD-best C (interleaved)

**Date**: 2026-07-18. Stage M5 of docs/duet/13. GPUs 0-4, ports
12900-12905, one run per cell, INTERLEAVED per B (duet_bB then c_bB).
All cells: ns=20 (×4 datasets = 80 prompts), out=256, in=512, temp 0.7,
seed 42, `--all`. DUET = champion E9K24_jit (m4-smoke args:
K1=9 [2×6,1×4], K2=4, exit 56, pfo=1, `SSD_FORCE_SPLIT_K1K2=1
SSD_DUET_JIT_SHORT=1`). C = `--k 7 --f 6`, no DUET gates.
All 6 cells rc=0, zero Tracebacks. Unrelated vLLM on GPUs 6-7 present
and unchanged at sweep start AND end (same regime throughout).

## Verdict up front

**The B>1 hypothesis (docs/duet/12 finding 5b) is REJECTED at v1**:
C scales better at every B. The gap widens −7.8% → −18.8% → −27.6%.
DUET's ingredient advantages all materialized (hit 0.84 vs 0.74 at B=4,
fewer rows/seq, cheaper misses) — and still lost, because (i) P2
dilution is a real B-effect (L_p2 1.64 → 0.49) that costs DUET ~10%
tokens/step while C's tok/step is flat, and (ii) DUET's step time grows
FASTER than C's on every axis (verify ×2.39 vs ×2.01, draft ×2.26 vs
×1.93) despite verifying half the rows per seq. Any-miss JIT stalls —
the mechanism the hypothesis leaned on — turn out NOT to be the growing
term for either system at B ≤ 4.

## Raw per-cell metrics

| metric | duet_b1 | c_b1 | duet_b2 | c_b2 | duet_b4 | c_b4 |
|---|---|---|---|---|---|---|
| Decode TPS (aggregate) | 71.86 | 77.90 | 89.22 | 109.86 | 108.87 | 150.31 |
| Tokens/step (incl recovery) | 3.62 | 3.96 | 3.24 | 3.90 | 3.27 | 3.99 |
| Cache hit rate (per-seq) | 0.80 | 0.74 | 0.82 | 0.73 | 0.84 | 0.74 |
| P1 (draft) hit rate | 0.523 | - | 0.428 | - | 0.392 | - |
| P2 (proxy) hit rate | 0.280 | - | 0.390 | - | 0.445 | - |
| L_p1 (P1 accepted len) | 3.54 | - | 4.05 | - | 5.07 | - |
| L_p2 (P2 accepted len) | 1.64 | - | 0.85 | - | 0.49 | - |
| Tok/step on hit | 3.88 | 4.23 | 3.52 | 4.26 | 3.62 | 4.29 |
| Tok/step on miss | 2.57 | 3.20 | 1.98 | 2.94 | 1.48 | 3.12 |
| T_target full step (ms) | 52.70 | 53.38 | 76.71 | 75.80 | 126.44 | 113.71 |
| T_verify (ms) | 46.41 | 45.41 | 66.39 | 58.26 | 110.86 | 91.42 |
| T_draft step (ms) | 45.09 | 39.05 | 65.29 | 62.89 | 102.06 | 75.28 |

Consistency check: hit·tok_hit + miss·tok_miss reproduces tok/step in
every cell (e.g. duet_b4: 0.84·3.62 + 0.16·1.48 = 3.28 ≈ 3.27;
c_b4: 0.74·4.29 + 0.26·3.12 = 3.99).

## 1. Scaling vs B

| B | DUET TPS | C TPS | DUET vs C | DUET ×B1 (eff.) | C ×B1 (eff.) | DUET /seq | C /seq |
|---|---|---|---|---|---|---|---|
| 1 | 71.86 | 77.90 | −7.8% | 1.00 | 1.00 | 71.9 | 77.9 |
| 2 | 89.22 | 109.86 | −18.8% | ×1.242 (62%) | ×1.410 (71%) | 44.6 | 54.9 |
| 4 | 108.87 | 150.31 | −27.6% | ×1.515 (38%) | ×1.930 (48%) | 27.2 | 37.6 |

Per-seq token latency (ms/tok): DUET 13.9 → 22.4 → 36.7;
C 12.8 → 18.2 → 26.6. C dominates on aggregate AND per-seq latency at
every B; each ×2 in B widens the gap ~10 points.

**Gap decomposition** (TPS ratio R = token ratio × step-time ratio;
step time t = B·tok_step/TPS, which matches T_target within ~5%):

| B | tok_D/tok_C | t_C/t_D | R = D/C |
|---|---|---|---|
| 1 | 0.914 | 1.008 | 0.921 |
| 2 | 0.831 | 0.978 | 0.813 |
| 4 | 0.820 | 0.884 | 0.725 |

B=1→2 the widening is almost purely TOKEN-side (P2 dilution, §3);
B=2→4 the TIME side joins (DUET's verify/draft growth, §2b).

## 2. Any-miss amplification — the hypothesis vs the data

Observed per-seq miss shares and the implied batch-level any-miss
burden P(any miss) = 1 − hit^B:

| B | DUET miss | C miss | DUET 1−h^B | C 1−h^B | DUET tok deficit/step | C tok deficit/step |
|---|---|---|---|---|---|---|
| 1 | 0.20 | 0.26 | 0.20 | 0.26 | 0.26 | 0.27 |
| 2 | 0.18 | 0.27 | 0.33 | 0.47 | 0.28 | 0.36 |
| 4 | 0.16 | 0.26 | 0.50 | 0.70 | 0.34 | 0.30 |

(tok deficit/step = miss_share · (tok_hit − tok_miss).)

DUET's hit advantage DID materialize and even widened (+0.06 → +0.10
absolute; theoretical any-miss burden 0.50 vs 0.70 at B=4). **It did
not pay.** Two measured reasons:

(a) **JIT stalls are not the growing per-step term at B ≤ 4.** If
step time were any-miss-coupled, C's step time (any-miss burden
0.26 → 0.70, +0.44) would have grown faster than DUET's (0.20 → 0.50,
+0.30). The opposite happened: C's T_target grew ×2.13 vs DUET's
×2.40, and C's tok/step is flat (3.96 → 3.99) — its higher miss burden
is fully amortized. The step-time growth that actually dominates is
batched GEMM physics: T_verify (D ×2.39 vs C ×2.01) and T_draft
(D ×2.26 vs C ×1.93). DUET grows faster on BOTH despite a 26-row/seq
budget vs C's 48 — consistent with (i) the v1 vk_max padding cost
(mixed batches always pay K1-width verify for all rows; at B=4
virtually every step has ≥1 long row), (ii) the DUET-only mid-verify
block (exit-56 proxy + batched Policy B + 2·B·wire_N wire) which was
measured never load-bearing at B=1 but is per-step serial work that
scales with B, and (iii) draft-side: 13 serial forwards whose B×16 /
B×10 rows cross the Marlin tile cliff at B ≥ 2 (the latency-bound
free-riding that made DUET cheap at B=1 inverts — C's 7 fat forwards
batch sublinearly, ×1.93 for ×4 rows, and its per-seq draft cost falls
52% vs DUET's 43%; slack T_target−T_draft at B=4: C 38.4 ms vs
DUET 24.4 ms).

(b) **Wrong currency, compounded** (finding 1 of docs/duet/12): a hit's
value is the avoided stall, not tokens — and DUET's misses, served by
K2=4 JIT-short, are token-capped. Tok/step on miss collapses with B
(2.57 → 1.98 → 1.48) while C's holds (3.20 → 2.94 → 3.12). At B=4
DUET loses MORE tokens/step to misses than C (0.34 vs 0.30) despite
10 points fewer misses. Cheap misses were a tail-latency advantage at
B=1; at B>1 they become a token liability.

Caveat on (a): SSD_PROFILE_DUET=0 for this sweep, so there is no
per-status step timing to split t_step(hit-only) vs t_step(any-miss)
directly; the attribution above rests on the cross-system growth-rate
comparison and the flat C tok/step, not on per-step timelines.

## 3. The M4 P2-composition flag — REAL B-effect, confirmed at ns=20

| B | P1 hit | P2 hit | L_p1 | L_p2 |
|---|---|---|---|---|
| 1 | 0.523 | 0.280 | 3.54 | 1.64 |
| 2 | 0.428 | 0.390 | 4.05 | 0.85 |
| 4 | 0.392 | 0.445 | 5.07 | 0.49 |

The M4 flag persists and is monotone in B at ns=20: the hit composition
shifts from draft-sourced to proxy-sourced (P2 share of hits 35% → 53%)
and P2 accepted length collapses 1.64 → 0.85 → 0.49 — at B=4 a
proxy-sourced hit is worth half a token. Magnitude note: M4's ns=8
tok/step −24% overstated it (ns noise); the real effect is ~−10%
tok/step at B=2 and B=4 (3.62 → 3.24/3.27) — but it is real, monotone,
and it alone accounts for the entire token side of the gap widening
(§1 decomposition), since C's tok/step is flat. Curiously L_p1 RISES
(3.54 → 5.07) while P1 hit rate falls — the surviving draft-sourced
hits are deeper. Mechanism unresolved (per-seq budget is by design
constant, so this is not a budget split across seqs; candidates:
off-policy staleness of the proxy hint growing with step time,
selection effects in which rows land P1 vs P2). Flagged for future
work.

## 4. Verdict + caveats

| claim (docs/duet/12 finding 5b) | measured |
|---|---|
| B>1 is DUET's structural home turf | **NO (v1)** — gap −7.8% → −27.6%, monotone in B |
| draft forwards ~free to batch (latency-bound) | NO at B≥2 — B×16 rows cross the tile cliff; T_draft ×2.26 vs C ×1.93 |
| 26 rows/seq batches deeper than 48 | verify rows/seq advantage did not show: T_verify ×2.39 vs C ×2.01 (vk_max padding + mid-verify block) |
| higher hit + cheap misses compound with B | hit advantage widened (0.84 vs 0.74) but any-miss stalls are not the binding term at B≤4; cheap misses became a token liability (1.48 tok on miss) |
| P2 composition flag (M4) | real B-effect: L_p2 1.64 → 0.49, P2 share of hits 35% → 53% |

What would have to change for DUET to win at B>1 (future work, in
expected-leverage order): (1) fix P2 dilution — it is the whole token
gap at B=2 (retune K2 / the P1-P2 budget split per B; understand the
composition shift first); (2) fewer, fatter draft forwards at B>1 —
the 13-forward deep-narrow shape is tuned to the B=1 tile cliff and
inverts at B≥2; (3) per-B verify dispatch (two-bucket or per-seq
width) to stop paying K1-width for short rows in mixed batches;
(4) move the mid-verify DUET block off the critical path (the B=1
null probes do not transfer to B>1).

Caveats: single run per cell (no error bars; the B=1 DUET-C gap of
−7.8% vs the +0.5% 5-rep headline shows out=256/ns=20 shifts the
baseline in C's favor — the headline convention is out=512/ns=50);
interleaved but not repeated (slow drift not controlled); unrelated
vLLM idle on GPUs 6-7 throughout (same regime for all cells, and
identical to the M1-M4 smoke regime); `--b 4` cells run 20 seqs in
waves of 4 (tail waves may be partially filled); SSD_PROFILE_DUET=0
(no per-status timing, see §2 caveat).

Repro: `run_all.sh` (this directory); per-cell logs in
`{duet,c}_b{1,2,4}/run.log`; `extract.py` regenerates the raw table.
