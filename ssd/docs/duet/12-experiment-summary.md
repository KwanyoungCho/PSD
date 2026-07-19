# 12 — DUET experiment summary (canonical index)

**Last updated**: 2026-07-19 (B>1 per-B shape sweep + confirmed wins added). One-page map of every experiment under
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

### B>1 recommended configs — per-B shape sweep, confirmed (2026-07-19)

Full sweep: `experiments/proxy_async_overlap/b_gt1/pb_sweep/RESULTS.md`
(14-cell K1/K2/dfo/pfo grid per B at ns=12, then 3-rep interleaved
confirm of each winner vs C at ns=20 out=256). **Recommended configs:**

| B | config | CLI shape | vs C (k7 f6) | confidence |
|---|---|---|---|---|
| 1 | E9K24_jit (champion, unchanged) | `--k 13 --f 3 --duet_phase1_k 9 --duet_phase2_k 4 --duet_draft_fan_out 2` + list `2,2,2,2,2,2,1,1,1,1` | +0.5% | 5-rep, 4/5 cycles, not band-clear (out=512 ns=50; −4.1% in the out=256 regime) |
| 2 | k6x5_d3p1 | `--k 11 --f 4 --duet_phase1_k 6 --duet_phase2_k 5 --duet_draft_fan_out 3` | **+6.9%** (114.09 vs 106.73) | **band-clear**, 3-rep interleaved: worst DUET 112.82 > best C 108.36 |
| 4 | k3x3_d4p1 | `--k 6 --f 5 --duet_phase1_k 3 --duet_phase2_k 3 --duet_draft_fan_out 4` | **+14.8%** (169.42 vs 147.53) | **band-clear**, 3-rep interleaved: worst DUET 167.24 > best C 151.28 |

(All with exit=56, policy b, jit-short, uniform phase-1 fan-out.)
The optimum gets shallower/fatter with B (K1 9 → 6 → 3, f 3 → 4 → 5):
verify-width cost scales with B (measured B×2.25 ms per K1 step) while
draft forwards stay latency-bound. Surface: K1 is the dominant knob at
B=4 (pure time effect; T_verify 60 → 87 ms for K1 3 → 6); K2=K1 adds
phase-2 tokens at zero verify cost (zero vk_max gap); pfo=2 is +2.7%
mid-grid and neutral at the winner; dfo is flat at B=4 but the main
B=2 knob (+3.6%). B=2 alternative k5x4_d3p1 (114.22 at scan) is
statistically indistinguishable from the confirmed winner.
**Finding 5b confidence statement**: the DUET-over-C win amplifies
with B (+0.5% → +6.9% → +14.8%), band-clear at B ∈ {2,4} — confirmed,
conditional on per-B shape retuning and the out=256/ns=20 regime;
K1=3 sits on the grid edge (K1=2 and B=8 unmeasured) and the B=4 win
is 100% step-time (tok/step 2.85 vs C 3.94).

## Removed implementations (git history registry)

| feature | gate | commits | why removed |
|---|---|---|---|
| Phase-2 hybrid / legacy two-pass | — | ≤ 19c8f73 | replaced by split-K1/K2 (2026-07) |
| KV promotion (glue removal) | `SSD_DUET_KV_PROMO` | 43d5b51..ad0b0ad, removed 10952bf | verified correct, but a wash at B=1 (docs/duet/11, kv_promo/RESULTS.md) |

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
