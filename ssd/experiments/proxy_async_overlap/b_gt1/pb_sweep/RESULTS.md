# pb_sweep — per-B DUET shape sweep + confirm (2026-07-18/19)

**Question** (follow-up to `../verdict/RESULTS.md` §4.3): fat5/fat7 were
the FIRST fat-shape guesses. Are K1/K2/dfo/pfo actually optimal per B?
Run a real grid per B ∈ {2,4}, then multi-rep-confirm the winner vs
SD-best C (k=7 f=6).

**Setup**: HEAD f543c24, GPUs 0-4 (target TP4 on 0-3, draft on 4),
in=512 out=256 temp 0.7 seed 42 `--all`, jit-short on, exit=56,
`SSD_FORCE_SPLIT_K1K2=1`, PROFILE=0. Scan: ns=12, one run/cell, ports
12930-12948 (`run_scan.sh` + `run_fixup.sh`). Confirm: ns=20, 3-rep
interleaved DUET/C per B, ports 12950+ (`run_confirm.sh`). Unrelated
vLLM idle on GPUs 6-7 throughout (same regime as m5/verdict).
Cell naming: `kAxB_dCpD` = K1=A, K2=B, dfo=C, pfo=D (k=K1+K2,
f=dfo+pfo, uniform phase-1 fan-out list [dfo]×(K1+1)).

## Verdict up front

**Both winners beat C band-clear (worst DUET rep > best C rep):**

- **B=4: k3x3_d4p1** (K1=3 K2=3 dfo=4 pfo=1, k=6 f=5) —
  **169.42 vs C 147.53 (+14.8%)**, spread 167.24-171.89 vs
  142.48-151.28.
- **B=2: k6x5_d3p1** (K1=6 K2=5 dfo=3 pfo=1, k=11 f=4) —
  **114.09 vs C 106.73 (+6.9%)**, spread 112.82-115.77 vs
  105.45-108.36.

With the B=1 headline (+0.5%, docs/duet/12), the win **amplifies with
B: +0.5% → +6.9% → +14.8%** — docs/duet/12 finding 5b (B>1 is DUET's
regime) is CONFIRMED, with the qualifier that the speculation shape
must be retuned per B (the deeper the batch, the shallower and fatter
the optimal shape). fat5 was NOT optimal at B=4: the scan's answer to
"is the shape optimal?" is **no** — the surface keeps rising as K1
drops to the grid edge (K1=3), +10.6% over fat5 at ns=12.

## 1. Scan tables (ns=12, one run/cell)

### 1a. B=4 grid (sorted by TPS; C anchor k7 f6)

| cell | k | f | TPS | tok/step | t_step (ms) | hit | P1/P2 hit | L_p1 | L_p2 | T_target | T_verify | T_draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k3x3_d4p2 | 6 | 6 | 166.27 | 2.84 | 68.3 | 0.89 | .750/.135 | 1.95 | 1.52 | 73.71 | 60.08 | 56.86 |
| **k3x3_d4p1** | 6 | 5 | **165.50** | 2.82 | 68.2 | 0.86 | .733/.129 | 2.00 | 1.44 | 73.59 | 60.11 | 53.66 |
| k4x4_d3p1 | 8 | 4 | 156.74 | 3.23 | 82.4 | 0.84 | .681/.163 | 2.51 | 1.62 | 86.74 | 71.40 | 69.07 |
| k4x3_d3p1 | 7 | 4 | 154.68 | 3.13 | 80.9 | 0.84 | .690/.150 | 2.46 | 1.48 | 86.10 | 71.56 | 64.47 |
| k5x4_d3p2 | 9 | 5 | 153.78 | 3.47 | 90.3 | 0.85 | .665/.189 | 2.86 | 1.74 | 96.88 | 80.36 | 81.08 |
| k6x5_d3p1 | 11 | 4 | 153.76 | 3.75 | 97.6 | 0.84 | .641/.194 | 3.22 | 1.99 | 105.20 | 87.08 | 86.72 |
| **C (k7 f6)** | 7 | 6 | 152.11 | 4.08 | 107.3 | 0.74 | — | — | — | 111.38 | 89.27 | 73.22 |
| k5x5_d3p1 | 10 | 4 | 149.89 | 3.46 | 92.3 | 0.83 | .654/.175 | 2.75 | 1.89 | 97.13 | 79.68 | 80.32 |
| k5x4_d3p1 (fat5) | 9 | 4 | 149.72 | 3.33 | 89.0 | 0.82 | .650/.170 | 2.75 | 1.59 | 94.77 | 78.52 | 75.43 |
| k5x4_d4p1 | 9 | 5 | 149.70 | 3.36 | 89.8 | 0.84 | .693/.146 | 2.66 | 1.81 | 95.94 | 79.80 | 76.01 |

### 1b. B=2 grid (sorted by TPS; C anchor k7 f6)

| cell | k | f | TPS | tok/step | t_step (ms) | hit | P1/P2 hit | L_p1 | L_p2 | T_target | T_verify | T_draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **k6x5_d3p1** | 11 | 4 | **114.35** | 3.63 | 63.5 | 0.84 | .653/.188 | 3.02 | 1.82 | 66.85 | 56.33 | 56.80 |
| k5x4_d3p1 | 9 | 4 | 114.22 | 3.41 | 59.7 | 0.84 | .665/.176 | 2.80 | 1.71 | 63.72 | 53.53 | 52.62 |
| k7x6_d2p1 | 13 | 3 | 112.33 | 3.76 | 66.9 | 0.81 | .557/.248 | 3.37 | 2.02 | 71.33 | 59.29 | 63.42 |
| k7x4_d2p1 (fat7) | 11 | 3 | 110.92 | 3.58 | 64.6 | 0.79 | .579/.214 | 3.26 | 1.69 | 68.82 | 58.27 | 55.69 |
| k6x5_d2p1 | 11 | 3 | 110.40 | 3.52 | 63.8 | 0.81 | .566/.239 | 3.02 | 1.88 | 67.46 | 56.54 | 57.34 |
| **C (k7 f6)** | 7 | 6 | 106.66 | 3.84 | 72.0 | 0.72 | — | — | — | 75.81 | 60.79 | 62.32 |

All 14 DUET cells + 2 C anchors: rc=0, zero Tracebacks — no cell hit a
config assert (the whole grid is inside the v1 constraint set K2≤K1,
dfo<f, k≤13).

## 2. The response surface story

**K1 (verify width) is the dominant knob at B=4, and it is a pure
time effect.** T_verify is a near-pure function of K1 alone:
60.1 (K1=3) / 71.4-71.6 (K1=4) / 78.5-80.4 (K1=5) / 87.1 (K1=6) ms —
+~9 ms per K1 step = B × 2.25 ms/row, exactly the marginal verify-row
cost the verdict profile measured (2.23 ms/row). K2 at fixed K1 moves
T_verify ≤1.2 ms. Each K1 step buys only ~+0.3 tok/step (~+9%) against
~+12% step time, so TPS climbs monotonically as K1 drops:
149.7 (K1=5) → ~155 (K1=4) → 165.5 (K1=3). The winner verifies
B×(K1+1) = 16 rows vs C's 32 — HALF of C's verify width and 44% faster
steps (67.4 vs 106.9 ms at ns=20) — while its hit rate rises to
0.86-0.89 (short chains are easy to hit; P1 hit .733 at K1=3 vs .650
at K1=5), holding the token ratio at 0.72 of C. Note the surface is
NOT globally monotone in K1: k6x5 (153.76) beats every K1=5 cell —
tokens (tok/step 3.75, L_p2 1.99) partially pay for depth again once
K2 tracks K1 — but it never threatens k3x3.

**K2=K1 (zero valid_k gap) is free tokens.** At fixed K1, closing the
K1−K2 gap adds phase-2 depth at ~zero verify cost (T_verify identical):
k4x4 156.74 > k4x3 154.68; k5x5 149.89 ≈ k5x4 149.72; at B=2,
k7x6 112.33 > k7x4 110.92. With K2=K1 every dispatch is the same width,
so the v1 vk_max padding term is structurally zero — the verdict's
"small K1−K2 gap neutralizes padding" mechanism taken to its limit.
The B=4 winner is a uniform-width K1=K2=3 shape.

**pfo is a real secondary knob mid-grid, neutral at the winner.**
k5x4_d3p2 vs _d3p1: +4.1 TPS (+2.7%) — +0.14 tok/step for only
+1.3 ms t_step; the extra proxy fan-out is funded by draft idle
(T_draft +5.7 ms but still under T_target), confirming the
"draft idle can fund pfo" hypothesis directionally. At the winner
shape the effect saturates: k3x3_d4p2 166.27 vs _d4p1 165.50 (+0.5%,
within single-run noise — the extra rows' token gain, 2.84 vs 2.82,
no longer covers their draft cost, T_draft +3.2 ms).

**dfo: flat at B=4, the main knob at B=2.** k5x4_d4p1 ≡ k5x4_d3p1
(149.70 vs 149.72). But at B=2 dfo 2→3 is worth +3.6% (k6x5: 114.35
vs 110.40) and both dfo=3 cells top the B=2 grid — at B=2 the draft
still has slack (phase-1 rows 2×dfo×(K1+1) stay near the Marlin
tiles) and fatter phase-1 lifts the hit rate 0.81 → 0.84.

**B=2 is a flat ridge, not a cliff.** The whole B=2 grid spans ±1.8%
(110.4-114.35) and every cell beats C_b2 (106.66) — at B=2 the verify
width term is half as steep (B×2.25 ms/row), so K1 5→6 is a wash
(114.22 vs 114.35, a tie at ns=12) and shape choice barely matters
beyond dfo=3. The winner call k6x5_d3p1 over k5x4_d3p1 is by 0.1% —
not resolvable at ns=12; we picked the higher-token cell for
robustness and confirmed IT, so the recommendation is "k6x5_d3p1
(confirmed) or k5x4_d3p1 (statistically indistinguishable at scan)".

## 3. Confirm phase (ns=20, 3-rep interleaved DUET/C per B)

### 3a. B=4 — k3x3_d4p1 vs C

| rep | DUET TPS | C TPS |
|---|---|---|
| r1 | 171.89 | 142.48 |
| r2 | 169.12 | 151.28 |
| r3 | 167.24 | 148.84 |
| **mean ± spread** | **169.42** (167.24-171.89) | **147.53** (142.48-151.28) |

**+14.8%, band-clear** (worst DUET 167.24 > best C 151.28 by +10.5%).
Mechanism (3-rep means): tok/step 2.85 vs 3.94 (ratio 0.723) ×
t_step 67.4 vs 106.9 ms (ratio 1.586) → R = 1.147 ✓. T_verify 60.5 vs
92.4 ms, T_draft 54.0 vs 75.0 ms, hit 0.87 vs 0.73 (any-miss burden
1−0.87⁴ ≈ 0.43 vs 0.72). DUET rep spread ±1.4%, C ±3.0%.

### 3b. B=2 — k6x5_d3p1 vs C

| rep | DUET TPS | C TPS |
|---|---|---|
| r1 | 113.67 | 108.36 |
| r2 | 112.82 | 106.39 |
| r3 | 115.77 | 105.45 |
| **mean ± spread** | **114.09** (112.82-115.77) | **106.73** (105.45-108.36) |

**+6.9%, band-clear** (worst DUET 112.82 > best C 108.36 by +4.1%).
Mechanism: tok/step 3.60 vs 3.79 (0.950) × t_step 63.1 vs 71.2
(1.128) → R = 1.071 ✓. T_verify 56.8 vs 58.4 (near-parity), T_draft
57.4 vs 62.6, hit 0.83 vs 0.73. Both sides' spreads ±1.5%.

All 12 confirm cells rc=0, zero Tracebacks.

### 3c. The amplification curve (finding 5b)

| B | best DUET shape | vs C | evidence |
|---|---|---|---|
| 1 | E9K24_jit (K1=9 K2=4, deep-narrow list) | **+0.5%** | 5-rep headline, docs/duet/12 (out=512 ns=50; −4.1% in the out=256 ns=20 regime) |
| 2 | k6x5_d3p1 (K1=6 K2=5 dfo=3 pfo=1) | **+6.9% band-clear** | 3-rep interleaved, this sweep |
| 4 | k3x3_d4p1 (K1=3 K2=3 dfo=4 pfo=1) | **+14.8% band-clear** | 3-rep interleaved, this sweep |

The optimal shape gets shallower and fatter as B grows (K1 9 → 6 → 3,
f 3 → 4 → 5) because the verify-width cost scales with B while draft
forwards stay latency-bound: B=1 optimizes tokens on the tile cliff,
B=4 optimizes step time on the verify GEMM. **Finding 5b is CONFIRMED
with per-B shape retuning** — B>1 is DUET's winning regime, and the
win grows with B.

## 4. Honest caveats

1. **Scan is one run/cell at ns=12.** Single-run noise at this length
   is ±3-4% (fat5 measured 149.72 here vs 155.12 in the verdict's
   ns=20 run). Mid-grid orderings (k4x4 vs k4x3, k6x5 vs k5x5, the
   B=2 top-two tie) are NOT resolvable; the winner margins (+9.5%
   over the next B=4 shape, +7-8 over the B=2 dfo=2 cells) are.
   Only the two confirmed winners carry multi-rep evidence.
2. **The C scan anchors and the confirm phase ran ~20h after the DUET
   scan cells** (the original scan's C cells crashed on a run-script
   argparse bug — stray positional arg, rc=2 before model load; fixed
   in `run_scan.sh`, rerun via `run_fixup.sh`). Cross-day drift could
   bias scan-table DUET-vs-C gaps, but not the confirm verdicts: the
   confirm phase is internally interleaved (DUET/C alternating,
   same session). C_b4 at ns=12 (152.11) and the confirm C_b4 mean
   (147.53) bracket the verdict's 150.31 — no regime shift visible.
3. **K1=3 sits on the grid edge.** The surface was still rising at the
   shallow end; K1=2 (verify 12 rows) is unmeasured, as is K2>K1
   (excluded by the v1 constraint) and B=8. k3x3_d4p2 (166.27, one
   run) suggests pfo=2 is at worst neutral at the winner shape —
   unconfirmed.
4. **Regime**: out=256 ns=20 temp 0.7, in=512, one prompt set (--all,
   seed 42). The B=1 headline (+0.5%) is from the out=512 ns=50
   regime; in THIS regime B=1 DUET measured −4.1% (m6_fix) — the B=1
   point of the amplification curve is regime-dependent, the B∈{2,4}
   points are not (measured here, interleaved).
5. **Token price**: the B=4 winner pays tok/step 2.85 vs C's 3.94 —
   the win is 100% step-time (67 vs 107 ms). Anything that raises
   per-token target cost (longer contexts, bigger models) shifts the
   frontier back toward deeper shapes; the per-B optimum is not
   universal.
6. GPUs 6-7 carried the same unrelated idle vLLM as all m5/verdict
   runs; GPUs 0-5 were otherwise free (verified at scan start).

Repro: `run_scan.sh` (grid), `run_fixup.sh` (C anchors + k3x3_d4p2),
`run_confirm.sh` (winners, env-parameterized), `extract.py` (tables);
raw run.log per cell in `<cell>/` and `confirm/<cell>/` (uncommitted,
standing prune policy docs/duet/12).
