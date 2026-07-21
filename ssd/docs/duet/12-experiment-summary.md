# 12 — DUET experiment summary (canonical index)

**Last updated**: 2026-07-21 (bscale32: B=16/32 extension + C-fairness re-verdict — the B>1 wins REVERSED against a per-B-optimized C). One-page map of every experiment under
`ssd/experiments/proxy_async_overlap/`, the verdict each one produced,
and where the details live. Raw profile JSONs (~3.9 GB) were pruned
2026-07-18; every distilled result survives in the per-directory
RESULTS.md files listed here. Reproduction: each directory keeps its
`run_all.sh` / `run_*.sh` with the exact CLI.

## Final configuration (champion)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4  # target TP4 on 0-3, draft on 4
SSD_FORCE_SPLIT_K1K2=1 SSD_DUET_JIT_SHORT=1 \
python -O bench.py --llama --size 8 <70B-AWQ + TinyLlama-AWQ args> \
  --async --spec --k 13 --f 3 --duet --duet_exit_layer 56 \
  --duet_phase1_k 9 --duet_phase2_k 4 --duet_draft_fan_out 2 \
  --duet_policy b --duet_split_phase1_fan_out_list 2,2,2,2,2,2,1,1,1,1
```

**Headline (5-rep interleaved, GPU 0-4, 2026-07-03)**:
DUET **81.91 ± 1.56** vs async-SD best (C: k=7 f=6) **81.52 ± 1.39** —
+0.5%, 4/5 cycles won, not band-clear. DUET reaches this with a
26-row speculation budget vs C's 48 rows; miss rate 0.18 vs 0.24 with
~25% cheaper misses (tail-latency advantage). Same-budget SD (D: f=3)
loses to DUET: 80.32 ± 1.67 vs 81.24 ± 0.67 (2026-07-02, 3-rep).

## Experiment index (chronological)

| # | directory | date | question | verdict |
|---|---|---|---|---|
| 1 | `phase0_baseline`, `phase0b`, `phase_b`, `phase_c` | 05 | timeline-profiling infra bring-up (aligned CUDA-event anchor, step_id wire, close-time status) | infra landed; measurement phases superseded by the final tooling |
| 2 | `async_sd_sweep`, `async_sd_sweep_7b_amd135m` | 05 | async-SD k×f grid → baseline operating points | **C = k7 f6 is SD's best**; D = k7 f3 is the budget-matched point |
| 3 | `mesa_k1k2_7_exit52_dfo_pfo_sweep` | 05 | early split-K1/K2 sweep (exit=52) | first 80+ TPS configs; superseded by exit=56 line |
| 4 | `k2_5_tps_verify` | 05-16 | does the paper config still hit ≥80 after engine WIP? | yes, 80.42; earlier "regression" was an output_len comparison artifact |
| 5 | `breakdown` | 05 | K1=K2=7 exit52 PROFILE breakdown | per-label baseline; representative-step plotting convention set |
| 6 | `p99_attribution` | 05 | where do p99 steps come from? | miss steps + JIT stalls dominate the tail |
| 7 | `batch3v`, `batch3v_70b` | 07-02 | async proxy send / proxy stream gates | **gain UNCONFIRMED** (peer-wait was already tiny); gates stay OFF |
| 8 | `b0_fix` | 07-02 | B=0 preempt crash fix validation | fixed (guard + split-aware lookahead); f=4-6 cells now run |
| 9 | `best_config_rematch` | 07-02 | A vs C vs D, 3-rep, post-cleanup | C 82.72±0.41 > A 81.24±0.67 (sig.); A ≈ D (tie); code changes caused no regression |
| 10 | `duet_wide_sweep` | 07-02 | widen dfo/pfo to close the C gap? | **REJECTED — substitution effect**: new P1 hits cannibalize P2, net tok +0.04 |
| 11 | `deep_p1_test` | 07-02 | deepen K1 (7→10)? | tokens CONFIRMED (geometric model exact) but **+3.4 ms/pos tax** → rejected as-is |
| 12 | `tax_decomposition` | 07-03 | decompose the 3.4 ms/pos | **draft Marlin tile cliff** (MQ 16→18: +44%/fwd) + spec_wait echo + ~2 ms/pos verify physics |
| 13 | `gate_scan` | 07-03 | JIT-short / proxy-on-draft gates + deep-narrow K1=8/9 | **JIT-short +3.85 (pure time)**; deep-narrow delivers depth tokens tile-free; E9_jit born |
| 14 | `final_rematch` | 07-03 | K2 sweep + pre-registered verdict + 7 probes | **champion E9K24_jit; +0.5% vs C (4/5)**; 5 target-side null probes + E10/pfo2 frontier probes |
| 15 | `champion_profile` | 07-04 | champion aligned timeline | per-status step anatomy; draft 45.6 vs target 51.4 (target-bound); PNGs kept |
| 16 | `kv_promo` | 07-04 | SwiftSpec-style glue removal (KV promotion) | **correct but a WASH** (glue span 5.44 = gather+tip 5.41 — batch-free forward); code REMOVED |
| 17 | `b_gt1` | 07-18 | B>1 support (M1-M4) + B ∈ {1,2,4} sweep vs C (M5) + verify-window bugfix (M6) | first M5 was bugged (short-row verify window); corrected: **B=2 near-parity (−4.8%)**, B=4 −21.5% and TIME-side only — finding 5b still unconfirmed but no longer token-broken |
| 18 | `b_gt1/verdict` | 07-18 | B=4 PROFILE forensics (bug or physics?) + fat-shape retune probes | **no remaining B>1 bug** (all labels match B×rows models); gap = vk_max padding 17-21 ms/step; **fat5 (K1=5 dfo=3) BEATS C at B=4: 155.12 vs 150.31 (+3.2%)** — first B>1 win, via shape retune |
| 19 | `b_gt1/pb_sweep` | 07-18/19 | per-B K1/K2/dfo/pfo grid (B∈{2,4}, 14 cells) + 3-rep confirm of each winner vs C | fat5 was NOT optimal; K1 (verify width) is the dominant knob; **B=4 k3x3_d4p1 +14.8% and B=2 k6x5_d3p1 +6.9% vs C, both BAND-CLEAR (3-rep interleaved)** — finding 5b CONFIRMED, win amplifies with B |
| 20 | `b_gt1/bscale` | 07-19 | B-scaling gap-fill: B=8 grid + B=4 K1=2 edge cells + B=1 same-regime anchors, then B=8 confirm | **B=8 k2x2_d5p1 (K1=K2=2 dfo=5) +26.9% vs C, BAND-CLEAR** (210.39 vs 165.85; worst rep beats best C by +23.7%); K1=3 is a real INTERIOR optimum at B=4 (K1=2 loses −5%); amplification curve complete +0.6/+6.9/+14.8/+26.9, shape law K1 9→6→3→2; REPORT.md + 5 figures in `bscale/figs/` **[SUPERSEDED by row 21: all vs-C numbers in this row used C fixed at k7f6]** |
| 21 | `b_gt1/bscale32` | 07-20/21 | B=16/32 extension + **C-fairness fix**: per-B optimize C too (31-cell C scan + 10-cell DUET scan + K1=1 probe + 5×3-rep interleaved confirms) | **REVERSAL — the B>1 band-clear wins were an artifact of the untuned k7f6 baseline.** Optimum-vs-optimum: DUET TIES at B∈{2,4} (+1.3%/−0.8%, overlap) and **LOSES band-clear at B∈{8,16,32}** (−3.7%/−2.5%/−4.1%); C's own shape law k 7→5→3→3→2→2 mirrors DUET's (same verify-width physics); per-B C gains vs k7f6 +9.7/+13.3/+35.8/+36.0%, and k7f6 is **DNF at B=32** (draft CG capture OOM, wall between 1152 and 1536 rows); K1=1 runs clean and wins DUET-internal at B=16/32 (law K1 9→6→3→2→1→1, 2→1 transition at B=16); `bscale32/REPORT.md` + 5 figures |

## The five load-bearing findings

1. **Wrong currency**: cache-hit gains don't carry tokens (P2 hit ≡
   pre-computed JIT — the user's identity, measured: tok/step 4.108 vs
   4.098 at hit 0.82 vs 0.76). A hit's only value is the avoided JIT
   stall, which JIT-short already halved.
2. **Tile cliff**: draft tree forwards are latency-bound up to the
   16-row Marlin m-tile; row 17 costs +44%/fwd. Deep-narrow fan-out
   lists (sum ≤ 16) buy depth for free → L_p1 3.60 → 4.25.
3. **JIT-short**: serving misses with a K2-deep JIT (valid_k=K2) cuts
   the JIT stall AND the miss's next verify width — the single biggest
   lever found (+3.85 tok/s).
4. **The frontier**: every verify position costs ~1.9 ms of coupled
   pipeline time ≈ the marginal token value at viable depths. Depth
   (E10), width (pfo=2, dfo), K2 — all slide ALONG the frontier; C sits
   on it. Five independent target-side probes (pod ×2, topm,
   replica ×2) proved the DUET-only mid-verify block is never
   load-bearing; draft-side savings (KV-promo, overlap) are absorbed by
   draft slack at B=1.
5. **Remaining levers** are (a) algorithmic — off-policy continuation
   quality L_p2 ≈ 1.8 vs breakeven 2.6 (draft adaptation), and
   (b) regime — DUET wins the draft-compute-bound settings (same-budget
   D loses; bigger draft / B>1 is DUET's ground). Target-side kernels
   (SwiftSpec) are Hopper sm_90 silicon — not portable to RTX 3090.
   **[Update 07-18: the B>1 half of (b) was measured (twice — the first
   sweep was invalidated by the M6 verify-window bug): with the B=1
   champion SHAPE, not a win at any B (B=2 near-parity, B=4 −21.5%,
   time-side). The verdict experiments then showed the shape was the
   whole story: with a B=4-appropriate fat shape (fat5: K1=5 dfo=3)
   DUET BEATS C 155.12 vs 150.31 (+3.2%, single run) — see the "B>1
   (2026-07-18)" section below.]**
   **[Update 07-19: CONFIRMED. The pb_sweep per-B grid found even
   better shapes (fat5 was not optimal), and 3-rep interleaved
   confirms are band-clear at both B: +6.9% (B=2) and +14.8% (B=4)
   over C — see "B>1 recommended configs" below.]**
   **[Update 07-19b: extended to B=8 — k2x2_d5p1 +26.9% band-clear.
   The complete amplification curve +0.6% → +6.9% → +14.8% → +26.9%
   (B = 1,2,4,8) and the shape law K1 9 → 6 → 3 → 2:
   `b_gt1/bscale/REPORT.md` + figures.]**
   **[Update 07-21 — REVERSED by the bscale32 fairness re-verdict.
   Every vs-C number above compared a per-B-retuned DUET against C
   FIXED at its B=1 optimum (k7f6). Giving C the same per-B shape
   optimization (its own law: k 7→5→3→3→2→2) erases the curve:
   DUET ties at B∈{2,4} (+1.3%/−0.8%, bands overlap) and loses
   band-clear at B∈{8,16,32} (−3.7%/−2.5%/−4.1%). B>1 throughput is
   therefore NOT DUET's winning regime on this hardware/regime; what
   remains of (b) is the untested draft-compute-bound and
   costlier-token settings, plus (a) unchanged. See
   `b_gt1/bscale32/REPORT.md`.]**

## B>1 (2026-07-18)

Design + staged implementation: docs/duet/13. Commits: design 7f30f36,
M1 baa011c (batched Policy B + B-axis wire + accept clamp), M2 af93cde
(vk_max dispatch + mixed hit/miss JIT-then-overwrite), M3 73fe75a
(batched selector + per-seq phase-2 fan-out/masks), M4 2cd2176 (gate
lift ≤8 + B=2 smoke), M6 = the verify-window bugfix (docs/duet/13 §M6).
Sweeps: `experiments/proxy_async_overlap/b_gt1/m5_sweep/` (first run —
DUET cells INVALID, kept as record) and `.../b_gt1/m6_fix/` (corrected
DUET cells, same args/GPUs).

**The M6 bug (found by auditing the M5 anomaly)**: the target verify
window `pos0 = num_tokens − (vk_max+1)` uses the batch-uniform vk_max,
but pre-M6 each seq's tokens were extended by its per-seq vk_i — every
SHORT row (P2 hit / JIT-short miss) in a MIXED batch verified against a
window slid (vk_max−vk_i)=5 tokens into known context. Its chain was
rejected against stale predictions (L_p2 → ~0.1 for affected rows) and
its recovery re-emitted old context tokens (silent output corruption —
the guard assert is stripped under `python -O`). Hit rate stayed HIGH
because keys are matched draft-side and corrupted rows churned bogus
P2 hits at position 0 (an attractor), which is why the bugged sweep
showed P2 hit 0.28→0.445 while L_p2 collapsed 1.64→0.49. Impossible at
B=1 (vk_i = vk_max), so all B=1 smokes had passed.

**Corrected key numbers** (aggregate decode TPS, DUET champion vs C):
B=1 74.69 vs 77.90 (−4.1%), B=2 104.59 vs 109.86 (**−4.8%,
near-parity**, tok/step 3.89 vs 3.90), B=4 118.00 vs 150.31 (−21.5%).
C scales ×1.93 B1→B4, DUET ×1.58. L_p2 (1.73/1.81/1.63), miss tokens
(2.59/2.71/2.68) and P2 hit rate (0.269/0.269/0.274) are B-invariant —
the "P2 dilution B-effect" and the "JIT-short token liability" of the
first sweep were bug artifacts, as was the curious L_p1-rise flag.

**Verdict experiments (2026-07-18, `b_gt1/verdict/RESULTS.md`) — no
remaining B>1 bug; the B=4 gap was the SHAPE, and a fat shape WINS.**
A B=4 profile run (champion args + SSD_PROFILE_DUET=1) checked every
label against its structural B×rows model: all 26 match (phase-1
per-forward 2.47→5.26 ms for 16→64 rows = 4 Marlin tiles, below the
tile-linear bound; batched JIT 8.6 vs 8.0 ms; per-seq mask builds
flat; walls fully label-accounted on both procs). Two structural
facts emerged: (i) the TARGET binds at B=4 — the draft's idle grew
6.0→34.5 ms/step, so the "13 serial forwards over the tile cliff"
never sit on the hit-step critical path; (ii) 93.3% of steps dispatch
K1-width verify while only 55% of rows are long → the v1 vk_max
padding costs **17-21 ms/step** (≈8.3 wasted rows × 2.23 ms/row),
the dominant time-side term. The miss-stall amplification term of
finding 5b IS present — any-miss burden 0.57 vs C's 0.70 (13-pt
frequency advantage, up from 6 pts at B=1) at 7.8 ms/stall — but
worth only +1..+5 ms/step, an order below the padding tax. Full
decomposition closes against the measured ΔT_target = +16.1 ms.

Retune probes (B=4, ns=20 out=256): **fat7** (K1=7 K2=4 uniform
dfo=2, k=11 — verify 32 rows = C's width, 11 forwards) 144.72 tok/s,
−3.7%, T_verify parity (91.97 vs 91.42), step FASTER than C;
**fat5** (K1=5 K2=4 dfo=3, k=9, --f 4 — verify 24 rows, 9 forwards)
**155.12 tok/s vs C 150.31 (+3.2%) — DUET's first measured B>1 win**,
trading tok/step 3.41 (0.855 of C) for step time 87.9 vs 106.2 ms
(hit 0.84, T_draft 75.5 ≈ C's 75.3). Finding 5b is thereby PARTIALLY
CONFIRMED at v1: B>1 is DUET's winning regime once the speculation
shape is retuned per B (fat, shallow) instead of inheriting the B=1
deep-narrow tile-cliff artifact.

**Remaining levers at B>1**: token side (tok/step 0.86-0.93 of C,
B-invariant — the L_p2 ≈ 1.7 off-policy continuation quality,
finding 5a); per-seq verify dispatch to reclaim residual padding in
mixed batches; a real per-B shape sweep (fat5 was the first guess;
fat5 at B ∈ {1,2} and K1 ∈ {4,5,6} × dfo grids are unmeasured — the
B=1 champion stays E9K24_jit). **[07-19: the per-B sweep was run —
next subsection. fat5 was indeed not optimal.]** Caveats: single run per cell (champion
B=4 tok/step spans 3.63-3.91 across identical-args runs, ±4% token
noise; +3.2% is not band-clear on its own — the robust result is
fat-beats-deep by +10..+31%); out=256/ns=20 shifts the B=1 baseline
in C's favor vs the out=512/ns=50 headline (+0.5% there, −4.1% here);
fat5 uses --f 4 (wider miss JIT); unrelated vLLM idle on GPUs 6-7,
unchanged across all cells; C cells were not re-run (no DUET code in
them).

### B>1 recommended configs — per-B shape sweep + B-scaling, confirmed (2026-07-19) **[SUPERSEDED 07-21 — see fairness re-verdict below]**

Full sweeps: `experiments/proxy_async_overlap/b_gt1/pb_sweep/RESULTS.md`
(14-cell K1/K2/dfo/pfo grid, B∈{2,4}, + 3-rep confirms) and
`.../b_gt1/bscale/REPORT.md` (B=8 grid + B=4 edge cells + B=1
same-regime anchors + B=8 confirm; the five campaign figures live in
`bscale/figs/` — fig1 TPS-vs-B, fig2 amplification curve, fig3 shape
law, fig4 B=4 response surface, fig5 per-seq latency).
**Recommended configs (historical — every vs-C column below compares
against C FIXED at k7f6):**

| B | config | CLI shape | vs C (k7 f6) | confidence |
|---|---|---|---|---|
| 1 | E9K24_jit (champion, unchanged) | `--k 13 --f 3 --duet_phase1_k 9 --duet_phase2_k 4 --duet_draft_fan_out 2` + list `2,2,2,2,2,2,1,1,1,1` | +0.5% | 5-rep, 4/5 cycles, not band-clear (out=512 ns=50; +0.6% single-run bscale anchor, −4.1% m6_fix, in the out=256 regime) |
| 2 | k6x5_d3p1 | `--k 11 --f 4 --duet_phase1_k 6 --duet_phase2_k 5 --duet_draft_fan_out 3` | **+6.9%** (114.09 vs 106.73) | **band-clear**, 3-rep interleaved: worst DUET 112.82 > best C 108.36 |
| 4 | k3x3_d4p1 | `--k 6 --f 5 --duet_phase1_k 3 --duet_phase2_k 3 --duet_draft_fan_out 4` | **+14.8%** (169.42 vs 147.53) | **band-clear**, 3-rep interleaved: worst DUET 167.24 > best C 151.28 |
| 8 | k2x2_d5p1 | `--k 4 --f 6 --duet_phase1_k 2 --duet_phase2_k 2 --duet_draft_fan_out 5` | **+26.9%** (210.39 vs 165.85) | **band-clear**, 3-rep interleaved: worst DUET 209.74 > best C 169.61 |

(All with exit=56, policy b, jit-short, uniform phase-1 fan-out.)
The optimum gets shallower/fatter with B — **K1 9 → 6 → 3 → 2,
f 3 → 4 → 5 → 6, verify rows/seq 10 → 7 → 4 → 3** — one K1 grid step
per doubling of B: verify-width cost scales with B (measured
B×2.25 ms per K1 step) while draft forwards stay latency-bound.
Surface: K1 is the dominant knob at B≥4 (pure time effect; T_verify
60 → 87 ms for K1 3 → 6 at B=4, 79.5 → 111.7 ms for K1 2 → 4 at B=8);
K2=K1 adds phase-2 tokens at zero verify cost (zero vk_max gap — every
winning B≥2 shape is uniform-width); pfo/dfo are ≤~1-3% trim near each
optimum. The bscale edge cells prove the law is not monotone: K1=2
LOSES at B=4 (157.3 vs 165.5), so K1=3 is a real interior optimum
there. B=2 alternative k5x4_d3p1 (114.22 at scan) and B=8 alternative
k2x2_d4p1 (211.61 at scan) are statistically indistinguishable from
the confirmed winners.
**Finding 5b confidence statement**: ~~the DUET-over-C win amplifies
with B (+0.5% → +6.9% → +14.8% → +26.9%), band-clear at B ∈ {2,4,8}~~
**[07-21: this statement did not survive the C-fairness re-verdict —
kept for history, see below]**. The B≥4 wins are 100%
step-time (B=8: tok/step 2.38 vs C 3.83, t_step 90 vs 185 ms; C's
any-miss burden 0.92 vs DUET 0.62); K1=1 and B>8 were the grid edges
closed by bscale32.

### B=1..32 fairness re-verdict + recommended configs, both systems (bscale32, 2026-07-21)

`experiments/proxy_async_overlap/b_gt1/bscale32/REPORT.md` (Korean;
RESULTS_scan.md + RESULTS_confirm.md + 5 figures in `bscale32/figs/`).
The campaign extended the curve to B=16/32 AND removed the fairness
gap: C was per-B optimized too (31-cell scan over k ∈ {2,3,5,7} ×
f ∈ {1,2,3,6}). **Optimum-vs-optimum, 3-rep interleaved confirms:**

| B | DUET-opt (shape) | C-opt (shape) | DUET TPS | C TPS | DUET vs C-opt | verdict |
|---|---|---|---|---|---|---|
| 1 | E9K24_jit | k7f6 | 72.24 | 71.80 | +0.6% | tie (single-run anchors) |
| 2 | k6x5_d3p1 | k5f6 | 115.73 | 114.24 | **+1.3%** | overlap — tie |
| 4 | k3x3_d4p1 | k3f6 | 168.09 | 169.43 | **−0.8%** | overlap — tie |
| 8 | k2x2_d5p1 | k3f6 | 210.21 | 218.30 | **−3.7%** | **C band-clear** |
| 16 | k1x1_d5p1 | k2f3 | 260.72 | 267.51 | **−2.5%** | **C band-clear** |
| 32 | k1x1_d4p1 | k2f2 | 288.95 | 301.19 | **−4.1%** | **C band-clear** |

**For B>1 throughput on this hardware/regime, the SD-best system is
per-B-optimized plain async-SD (C-opt), not DUET.** C obeys the same
shape law DUET does (k* 7→5→3→3→2→2 vs K1* 9→6→3→2→1→1 — both ride
the verify-width frontier; C's f* also collapses 6→3→2 at B≥16 under
the draft-CG memory pressure). Per-B optimizing C gains +9.7/+13.3/
+35.8/+36.0% over fixed k7f6 at B=2/4/8/16, and at B=32 fixed k7f6 is
**DNF** (draft CG capture OOM at 1536 rows; the 24 GB wall is between
1152 and 1536 rows — `(k+1)×f×B`). What survives for DUET: B=1
champion parity, the K1=1 discovery (first run ever; wins
DUET-internal at B=16/32; the K1 2→1 transition is at B=16 — the B=8
probe lost 209.07 vs 213.51), cross-campaign reproducibility (B=8
DUET 210.21 vs 210.39), and the untested draft-compute-bound /
costlier-token regimes. Mechanism of the residual B≥8 gap: DUET still
wins step time (B=32 t_step 197 vs 253 ms; verify rows 64 vs 96) but
loses tokens (tok/step 1.78 vs 2.38 — the phase-2 off-policy
continuation L_p2 0.62 is worth less than C's on-policy chain
position), and the hit-rate edge (0.90 vs 0.71) stops paying at large
B because the any-miss burden saturates for both (1−hit^B ≈ 0.97 vs
1.00 at B=32).

## Removed implementations (git history registry)

| feature | gate | commits | why removed |
|---|---|---|---|
| Phase-2 hybrid / legacy two-pass | — | ≤ 19c8f73 | replaced by split-K1/K2 (2026-07) |
| KV promotion (glue removal) | `SSD_DUET_KV_PROMO` | 80eb896..41bee95, removed c427a72 | verified correct, but a wash at B=1 (docs/duet/11, kv_promo/RESULTS.md) |

Live optional gates (default OFF, kept): `SSD_DUET_JIT_SHORT`
(**champion uses it**), `SSD_DUET_PROXY_ON_DRAFT` (+`SSD_DUET_PROXY_TOPM`),
`SSD_DUET_EXIT_TOPM_GATHER`, `SSD_DUET_EXIT_REPLICA`,
`SSD_ASYNC_PROXY_SEND` (+`SSD_PROXY_STREAM`) — all measured neutral at
B=1; candidates for the draft-bound-regime work or future pruning.

## What was pruned on 2026-07-18

- All raw `mesa_profile_*.json` / `duet_profile_*.json` dumps (~3.8 GB;
  the champion's timeline PNGs + all RESULTS.md tables survive; any
  profile is reproducible from the kept run scripts in ~10-25 min).
- `kv_promo/smoke_*` debug runs; `final_rematch/altset_partial/`
  (contaminated GPU-set partials, documented in final_rematch/RESULTS.md).
