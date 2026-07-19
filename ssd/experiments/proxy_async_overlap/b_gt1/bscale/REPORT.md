# bscale — the B-scaling story, B ∈ {1,2,4,8} (2026-07-19)

**Question** (closes pb_sweep/RESULTS.md caveat 3): the per-B sweep left
K1=3 on the B=4 grid edge and B=8 unmeasured, with the optimum trend
K1 9 → 6 → 3 extrapolating to K1 ≈ 2-3 at B=8. Does the DUET-over-C
amplification continue at B=8, is K1=3 a real interior optimum at B=4,
and what is the complete per-B shape law?

**Setup**: HEAD d0be348, GPUs 0-4 (target TP4 on 0-3, draft on 4),
in=512 out=256 temp 0.7 seed 42 `--all`, jit-short on, exit=56,
`SSD_FORCE_SPLIT_K1K2=1`, PROFILE=0, uniform phase-1 fan-out
(`[dfo]×(K1+1)`), C = async-SD best (k=7 f=6). Scan: ns=12, one
run/cell, ports 12970+ (`run_scan.sh`). Confirm: ns=20, 3-rep
interleaved DUET/C, ports 13000+ (`run_confirm.sh`). Unrelated vLLM
idle on GPUs 6-7 throughout (same regime as all pb_sweep/verdict runs);
GPUs 0-5 otherwise free at scan start. Cell naming: `b<B>_kAxB_dCpD` =
K1=A, K2=B, dfo=C, pfo=D (k=K1+K2, f=dfo+pfo).

## Verdict up front

**B=8: k2x2_d5p1 (K1=K2=2, dfo=5 pfo=1, k=4 f=6) beats C band-clear —
210.39 vs 165.85 (+26.9%)**, spread 209.74-211.11 vs 162.64-169.61
(worst DUET rep beats best C rep by +23.7%). The full amplification
curve is now measured at every power of two:

**+0.6% (B=1) → +6.9% (B=2) → +14.8% (B=4) → +26.9% (B=8)**,
band-clear at B ∈ {2,4,8} — docs/duet/12 finding 5b holds through B=8
and keeps growing. The optimal shape keeps getting shallower and
fatter: **K1 9 → 6 → 3 → 2**, f 3 → 4 → 5 → 6, verify rows/seq
(K1+1) 10 → 7 → 4 → 3. And the B=4 edge cells show the law is not
"smaller is always better": K1=2 LOSES at B=4 (157.3 < 165.5), so
K1=3 is a genuine interior optimum there — the K1 frontier slides with
B, one step per doubling.

## 1. Phase A — gap-filling scan (ns=12, one run/cell)

Completed-cell checklist (planned 11 / completed 11, all rc=0, zero
Tracebacks; no cell hit a config assert — B=8 is inside the v1
constraint set, M4 gate ≤ 8):

| planned cell | done | TPS |
|---|---|---|
| b8_k2x2_d4p1 | yes | 211.61 |
| b8_k2x2_d5p1 | yes | **213.51** (winner) |
| b8_k3x3_d4p1 | yes | 207.92 |
| b8_k3x3_d4p2 | yes | 209.66 |
| b8_k4x4_d3p1 | yes | 189.73 |
| b8_c (k7 f6) | yes | 163.21 |
| b4_k2x2_d4p1 | yes | 157.28 |
| b4_k2x2_d5p1 | yes | 151.79 |
| b4_k3x3_d5p1 | yes | 165.45 |
| b1_e9k24_jit | yes | 72.24 |
| b1_c (k7 f6) | yes | 71.80 |

### 1a. B=8 grid (sorted by TPS; C anchor k7 f6)

| cell | k | f | TPS | tok/step | t_step (ms) | hit | P1/P2 hit | L_p1 | L_p2 | T_target | T_verify | T_draft |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **k2x2_d5p1** | 4 | 6 | **213.51** | 2.40 | 89.9 | 0.90 | .812/.087 | 1.48 | 1.08 | 98.85 | 79.49 | 67.52 |
| k2x2_d4p1 | 4 | 5 | 211.61 | 2.38 | 90.0 | 0.88 | .780/.105 | 1.47 | 1.04 | 98.95 | 79.60 | 66.52 |
| k3x3_d4p2 | 6 | 6 | 209.66 | 2.88 | 109.9 | 0.89 | .768/.126 | 2.03 | 1.40 | 112.62 | 91.05 | 85.09 |
| k3x3_d4p1 | 6 | 5 | 207.92 | 2.83 | 108.9 | 0.86 | .742/.117 | 1.98 | 1.49 | 111.37 | 89.33 | 79.55 |
| k4x4_d3p1 | 8 | 4 | 189.73 | 3.19 | 134.5 | 0.82 | .664/.152 | 2.44 | 1.69 | 137.31 | 111.67 | 104.53 |
| **C (k7 f6)** | 7 | 6 | 163.21 | 3.85 | 188.7 | 0.72 | — | — | — | 191.15 | 156.08 | 115.09 |

The pb_sweep physics carries straight to B=8: T_verify is again a
near-pure function of K1 (79.5 / ~90 / 111.7 ms for K1 = 2/3/4; C's
k=7 verify pays 156 ms for 64 rows), every DUET cell beats C by
+16..+31%, and the whole K1 ∈ {2,3} block sits within 2.7% (the
k2x2_d5p1-over-d4p1 margin, +0.9%, is inside single-run noise — the
scan's job was to pick a confirm candidate, and any of the top four
would have carried the verdict). dfo=5 at K1=2 keeps phase-1 rows at
dfo×(K1+1) = 15/seq and lifts P1 hit to .812; hit rate 0.90 vs C's
0.72.

### 1b. B=4 edge cells (with the pb_sweep neighbors for context)

| cell | k | f | TPS | tok/step | t_step (ms) | hit | L_p1 | L_p2 | T_verify | source |
|---|---|---|---|---|---|---|---|---|---|---|
| k3x3_d4p2 | 6 | 6 | 166.27 | 2.84 | 68.3 | 0.89 | 1.95 | 1.52 | 60.08 | pb_sweep |
| k3x3_d4p1 | 6 | 5 | 165.50 | 2.82 | 68.2 | 0.86 | 2.00 | 1.44 | 60.11 | pb_sweep (confirmed winner) |
| **k3x3_d5p1** | 6 | 6 | 165.45 | 2.83 | 68.4 | 0.87 | 1.95 | 1.42 | 59.33 | **bscale** |
| **k2x2_d4p1** | 4 | 5 | 157.28 | 2.41 | 61.3 | 0.90 | 1.51 | 0.93 | 53.58 | **bscale** |
| **k2x2_d5p1** | 4 | 6 | 151.79 | 2.30 | 60.6 | 0.88 | 1.38 | 0.96 | 53.96 | **bscale** |
| C (k7 f6) | 7 | 6 | 152.11 | 4.08 | 107.3 | 0.74 | — | — | 89.27 | pb_sweep |

**K1=3 at B=4 is an interior optimum, not a grid-edge artifact.** The
K1 3 → 2 step saves only 7 ms of step time (68.2 → 61.3, −10%) but
costs 0.41 tok/step (2.82 → 2.41, −15%) — at B=4 the token loss wins
and the surface falls 165.5 → 157.3. At B=8 the same step saves 19 ms
(108.9 → 89.9, −17%) against the same −15% tokens — there it flips
positive. That is the whole shape law in one comparison: **the verify
width term scales with B (measured ≈ B × 2.25 ms per K1 step), the
token value of depth does not, so the optimal K1 drops by one grid
step per doubling of B.** No new B=4 winner (all three edge cells
< 165.5), so no B=4 re-confirm was needed. Also dfo 4→5 at k3x3 is
neutral (165.45 ≈ 165.50), same saturation the pfo probe showed.

### 1c. B=1 same-regime anchors (single run, ns=12)

| cell | TPS | tok/step | t_step (ms) | hit | T_target | T_draft |
|---|---|---|---|---|---|---|
| E9K24_jit (champion) | 72.24 | 3.61 | 50.0 | 0.82 | 52.49 | 44.92 |
| C (k7 f6) | 71.80 | 3.74 | 52.1 | 0.71 | 54.57 | 39.20 |

+0.6% — consistent with the 5-rep out=512 headline (+0.5%) and, being
single-run, consistent with parity. These anchors exist so the B=1
point of the curve is measured in the SAME ns/out/day regime as the
rest of this campaign (the m6_fix ns=20 B=1 point was −4.1%; B=1 sits
at parity either way, and the amplification story does not lean on it).

## 2. Phase B — B=8 confirm (ns=20, 3-rep interleaved DUET/C)

Completed cells 6/6 (b8_duet_r1..3, b8_c_r1..3), all rc=0, zero
Tracebacks, ports 13000-13007.

| rep | DUET k2x2_d5p1 TPS | C TPS |
|---|---|---|
| r1 | 211.11 | 165.31 |
| r2 | 210.33 | 169.61 |
| r3 | 209.74 | 162.64 |
| **mean ± spread** | **210.39** (209.74-211.11) | **165.85** (162.64-169.61) |

**+26.9%, band-clear** (worst DUET 209.74 > best C 169.61 by +23.7%).
DUET rep spread ±0.3% (remarkably tight), C ±2.1%.

Mechanism (3-rep means): tok/step 2.38 vs 3.83 (ratio 0.621) ×
t_step 90.4 vs 184.7 ms (ratio 2.044) → R = 1.269 ✓. The win is 100%
step-time, bought at the verify GEMM: DUET verifies B×(K1+1) = **24
rows vs C's 64** → T_verify 80.7 vs 160.1 ms; T_draft 67.9 vs 117.7
(both sides target-bound). Hit rate 0.89 vs 0.73 — the any-miss burden
1−hit^B is 0.62 vs C's **0.92**: at B=8, C runs a JIT-degraded step
92% of the time. C is visibly saturating on the width axis: its
aggregate gain B=4→8 is only +12.4% (147.53 → 165.85, step time
almost doubles, 106.9 → 184.7 ms) while DUET's is +24.2%
(169.42 → 210.39). Within the scan (same day, same ns), B=1→8
aggregate scaling is DUET ×2.96 vs C ×2.27.

## 3. The complete amplification curve (finding 5b, final)

| B | best DUET shape | k | f | verify rows/seq | DUET TPS | C TPS | vs C | evidence |
|---|---|---|---|---|---|---|---|---|
| 1 | E9K24_jit (K1=9 K2=4, list [2×6,1×4]) | 13 | 3 | 10 | 81.91 | 81.52 | **+0.5%** | 5-rep interleaved, out=512 ns=50 (docs/duet/12); same-regime ns=12 anchor: +0.6% single-run |
| 2 | k6x5_d3p1 (K1=6 K2=5 dfo=3) | 11 | 4 | 7 | 114.09 | 106.73 | **+6.9% band-clear** | 3-rep interleaved, pb_sweep |
| 4 | k3x3_d4p1 (K1=3 K2=3 dfo=4) | 6 | 5 | 4 | 169.42 | 147.53 | **+14.8% band-clear** | 3-rep interleaved, pb_sweep |
| 8 | k2x2_d5p1 (K1=2 K2=2 dfo=5) | 4 | 6 | 3 | 210.39 | 165.85 | **+26.9% band-clear** | 3-rep interleaved, this experiment |

The advantage roughly doubles per doubling of B once B > 1. Every
winner at B ≥ 2 is a uniform-width K1=K2 shape (zero vk_max padding —
the verdict's dominant B>1 cost term engineered to zero by shape
choice), and every step down in K1 is paid for by B growing the
verify-width cost while draft forwards stay latency-bound.

## 4. Figures

![fig1](figs/fig1_tps_vs_B.png)

**Fig 1 — aggregate decode TPS vs B.** DUET (per-B best shape) scales
near-linearly on the log2 axis through B=8 while C visibly bends after
B=4: C's 64-row verify at B=8 nearly doubles its step time for only
+12% throughput, whereas DUET re-buys headroom at every B by shrinking
K1. Error bars (min/max over the 3-rep interleaved confirms) are
smaller than the markers on the DUET side at B=8 (±0.3%); B=1 points
are single-run same-regime anchors at parity.

![fig2](figs/fig2_advantage_vs_B.png)

**Fig 2 — the amplification curve.** The DUET-over-C advantage
+0.6% → +6.9% → +14.8% → +26.9% for B = 1 → 2 → 4 → 8, band-clear
(worst DUET rep > best C rep, 3-rep interleaved) at every B ≥ 2. This
is docs/duet/12 finding 5b measured to completion: B>1 is DUET's
regime, and the win compounds — roughly doubling per doubling of B —
conditional on retuning the speculation shape per B.

![fig3](figs/fig3_optimal_shape_vs_B.png)

**Fig 3 — the per-B shape law.** Optimal K1 falls one grid step per
doubling of B (9 → 6 → 3 → 2), K2 converges onto K1 (uniform width =
zero vk_max padding), verify rows/seq (K1+1) collapse 10 → 3, and the
fan-out f rises 3 → 6 to spend the freed draft budget on width instead
of depth. B=1 optimizes tokens on the draft tile cliff (deep-narrow);
B=8 optimizes step time on the target verify GEMM (shallow-fat).

![fig4](figs/fig4_b4_response_surface.png)

**Fig 4 — the B=4 response surface, now with both edges.** All 12 B=4
DUET scan cells (pb_sweep grid + bscale K1=2 edge cells) vs K1. The
surface rises monotonically K1 6 → 3 and then FALLS at K1=2
(157.3/151.8 < 165.5): K1=3 is an interior optimum, closing
pb_sweep's grid-edge caveat. Color (ordinal blue ramp) is dfo, marker
is pfo; the dfo/pfo choice moves cells ≤ ~1% near the optimum — K1 is
the dominant knob, everything else is trim.

![fig5](figs/fig5_per_seq_latency.png)

**Fig 5 — the throughput/latency tradeoff.** Per-seq token rate
(aggregate/B) falls with B for both systems — batching is not free —
but DUET's curve sits above C's at every B > 1 and the gap widens:
26.3 vs 20.7 tok/s/seq at B=8 (+27%). Same runs as fig 1. Reading it
as a serving frontier: for any per-seq latency target below ~70 tok/s,
DUET reaches it at a higher batch size (more aggregate throughput)
than C.

## 5. Recommended per-B configs (final)

All with `--async --spec --duet --duet_exit_layer 56 --duet_policy b`,
`SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1`, uniform phase-1 fan-out
unless a list is given:

| B | config | CLI shape | vs C (k7 f6) | confidence |
|---|---|---|---|---|
| 1 | E9K24_jit | `--k 13 --f 3 --duet_phase1_k 9 --duet_phase2_k 4 --duet_draft_fan_out 2 --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1` | +0.5% | 5-rep, 4/5 cycles, not band-clear |
| 2 | k6x5_d3p1 | `--k 11 --f 4 --duet_phase1_k 6 --duet_phase2_k 5 --duet_draft_fan_out 3` | **+6.9%** | band-clear, 3-rep |
| 4 | k3x3_d4p1 | `--k 6 --f 5 --duet_phase1_k 3 --duet_phase2_k 3 --duet_draft_fan_out 4` | **+14.8%** | band-clear, 3-rep |
| 8 | k2x2_d5p1 | `--k 4 --f 6 --duet_phase1_k 2 --duet_phase2_k 2 --duet_draft_fan_out 5` | **+26.9%** | band-clear, 3-rep |

Rule of thumb for unswept B: K1 = K2 ≈ max(2, 9/B rounded to the grid),
f = dfo+1 with dfo filling the draft's idle budget; when in doubt,
prefer the shallower shape — the surface falls gently on the shallow
side (−5% at B=4) and steeply on the deep side (−12..−24%).

## 6. Mechanism summary

1. **Verify width dominates B-scaling.** Marginal verify cost is
   ≈ 2.25 ms per row × B (measured at B=4 in pb_sweep, consistent at
   B=8: T_verify 79.5 → 111.7 ms for K1 2 → 4 ≈ 16 ms per K1 step =
   8 × 2.0). C's fixed k=7 (8 rows/seq) becomes 64 verify rows at B=8
   → 160 ms of verify per step; DUET buys its throughput back by
   verifying 3 rows/seq. The win is 100% step-time; DUET pays
   tok/step 2.38 vs C's 3.83 for it.
2. **Hit advantage compounds with B.** DUET 0.89 vs C 0.73 per-seq
   hit rate → any-miss burden at B=8 of 0.62 vs 0.92 (and DUET's
   misses are JIT-short, cheaper). This is the M2 mixed hit/miss
   design doing its job — one miss no longer clobbers seven hits.
3. **K1=K2 (uniform width) at every winning B ≥ 2** — the v1 vk_max
   padding term (17-21 ms/step at B=4 with the deep champion) is
   structurally zero.
4. **Fan-out is the residual knob**: draft idle grows with B on the
   winner shapes (draft 67.9 vs target 99.2 ms at B=8), so fatter
   phase-1 (dfo 3 → 4 → 5) is free hit-rate (P1 hit .81 at the B=8
   winner); its effect saturates ≤ ~1% near each optimum.
5. **The shape law**: optimal K1 halves-ish per B doubling
   (9 → 6 → 3 → 2), because depth's token value is B-invariant while
   width's time cost is linear in B. Off-policy continuation quality
   (L_p2 ≈ 1.1-1.7, finding 5a) remains the open token-side lever.

## 7. Caveats

1. **Scan is one run/cell at ns=12** (±3-4% single-run noise): the
   B=8 top-four ordering (213.5/211.6/209.7/207.9) is not resolvable —
   only the confirm verdict (k2x2_d5p1 vs C, band-clear) is load-
   bearing. k2x2_d4p1 is statistically indistinguishable from the
   confirmed winner.
2. **ns is not a multiple of 8 at B=8**: 12 seqs = one full 8-batch +
   a 4-seq tail (scan); 20 = 8+8+4 (confirm). Tail steps run below
   full width, so absolute B=8 numbers blend in some narrower-batch
   steps. DUET and C see the identical admission pattern (interleaved,
   same ns/seed), so the vs-C verdicts are unaffected.
3. **K1=2 is the new grid edge at B=8** (K1=1, i.e. 2-row verify, is
   unmeasured; so is B>8 — the v1 gate caps max_num_seqs at 8, and
   the KV pool at 2048 tokens/seq would bind soon after). K2>K1 stays
   excluded by the v1 constraint.
4. **Regime**: out=256, in=512, temp 0.7, one prompt set (--all,
   seed 42), ns=12/20. The B=1 point is regime-sensitive (+0.5%
   out=512 headline, +0.6% here single-run, −4.1% in m6_fix ns=20);
   the B ∈ {2,4,8} band-clear verdicts are all in-regime and
   internally interleaved.
5. **Token price**: the B=8 winner accepts 0.62 of C's tokens/step —
   anything that raises per-token target cost (longer contexts, bigger
   models, costlier sampling) shifts the per-B optimum back toward
   depth; the shape law's SLOPE (shallower with B) should survive,
   the absolute K1 values may not.
6. **Cross-day drift**: B=2/B=4 confirm numbers are from pb_sweep
   (2026-07-19, same GPUs/regime); bscale ran ~19-20h later. Each
   confirm is internally interleaved, so each per-B verdict is
   drift-safe; only cross-B comparisons of absolute TPS carry the
   usual day-to-day caveat. Same unrelated idle vLLM on GPUs 6-7
   throughout both.

Repro: `run_scan.sh` (11 cells), `run_confirm.sh` (env-parameterized
winner, `B8_K1=2 B8_K2=2 B8_DFO=5 B8_PFO=1`), `extract.py` (tables),
`plot_figs.py` (figs/, parses run.logs directly); raw run.log per cell
in `<cell>/` and `confirm/<cell>/` (uncommitted, standing prune policy
docs/duet/12). pb_sweep data: `../pb_sweep/RESULTS.md`.
