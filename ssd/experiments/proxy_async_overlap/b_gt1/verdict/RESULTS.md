# Verdict experiments — is the B=4 gap a bug or physics? (2026-07-18)

**Question** (follow-up to m5_sweep/RESULTS.md corrected verdict and
docs/duet/13 §M6): after the M6 verify-window fix, DUET at B=4 still
loses −21.5% to SD-best C (118.00 vs 150.31). Prove or refute: "there is
no remaining B>1 bug — the gap is draft-shape + vk_max-padding time
costs", and produce a causal decomposition.

**Setup**: HEAD 9528366, GPUs 0-4 (target TP4 on 0-3, draft on 4),
ns=20 out=256 in=512 temp 0.7 seed 42, `--all`, B=4. Unrelated vLLM
idle on GPUs 6-7 throughout (same regime as m5/m6).

- **Exp1** `prof_b4/`: corrected champion (E9K24_jit, m6_fix duet_b4
  args) + `SSD_PROFILE_DUET=1`, port 12920. rc=0, 126.69 tok/s.
- **Exp2** `fat7/`, `fat5/`: B>1-shape retune probes, PROFILE=0, ports
  12921-2 (see §3). Both rc=0, zero Tracebacks.
- Analysis: `analyze_prof.py` (this dir; full tables in
  `analyze_prof_out.md`), plus `tax_decomposition/analyze.py --base
  ../b_gt1/verdict --cells prof_b4:4` (`analyze_tax_prof_b4.md`).

## Verdict up front

**No remaining B>1 bug.** Every profile label matches its structural
row model within noise (§1). The champion-shape B=4 gap decomposes as:
**vk_max-padded verify width ≈ 17-21 ms/step (the dominant term, ~2/3
of the time-side gap incl. its knock-ons) + the pre-existing ~−7%
token deficit (B-invariant, same as B=1) − a real but small miss-stall
advantage (+1..+5 ms/step in DUET's favor)**. The "13 serial draft
forwards over the tile cliff" story is REFUTED as the binding cost:
the draft has MORE slack at B=4 (34.5 ms idle) than at B=1 (6.0 ms) —
the target verify GEMM grew faster than the draft did.

Measured proof (Exp2): cutting K1 9→7 (fat7: verify width 40→32 rows =
C's exact width, 13→11 forwards) recovers **+22.6% TPS (118.00 →
144.72, −3.7% vs C)** — T_verify lands at 91.97 ms vs C's 91.42
(parity) and DUET's full step becomes FASTER than C's (t=102.5 vs
106.2 ms). Cutting further (fat5: K1=5, verify 24 rows, 9 forwards)
**BEATS C: 155.12 vs 150.31 (+3.2%)** — DUET's first B>1 win,
trading −14.5% tokens for −17% step time. **fat5 is the B=4 config**
(single-run caveat §4).

## 1. Exp1 — profile forensics vs the structural ROW MODEL

1340 profiled steps (1240 post-warmup; 1229 full-batch). Status shares
(draft): any-miss (mixed+miss) 56.5%, all-hit 42.7%, ramp 0.9% —
any-miss share matches 1−hit^B = 1−0.81^4 = 0.57 exactly.

### 1a. Draft labels (any-miss steps, n=699) vs model

| label | B=1 (hit_k1) | B=4 | model expectation (64/40-row rows, 4/3 Marlin m-tiles) | verdict |
|---|---|---|---|---|
| phase1_replay | 22.58 (9×2.51) | 47.31 (9×5.26) | 9 × ≤5.79 (tile-linear bound 2.52+3×1.09) | ✓ below bound (×2.13 for ×4 rows) |
| phase2_replay | 9.93 (4×2.48) | 17.81 (4×4.45) | 4 × ≤4.70 (3 tiles) | ✓ |
| draft_glue_replay | 1.78 | 3.60 | 40 rows vs 10 (3 tiles) | ✓ (×2.0) |
| glue (incl build) | 2.70 | 4.48 | — | ✓ |
| phase1_prep | 3.23 | 4.04 | CPU prep, 9 units | ✓ (+0.8 total) |
| phase2_prep | 1.69 | 1.95 | CPU prep, 4 units | ✓ |
| phase1_build / phase2_build | 0.78 / 0.85 | 1.00 / 0.83 | per-seq nested fan_out_list mask build (M3) | ✓ flat — no rebuild blowup |
| merge_cache | 0.11 | 0.32 | B× keys | ✓ tiny |
| draft_send_response | 0.22 | 0.47 | 4× wire | ✓ tiny |
| hit_cache_respond (all-hit) | 0.89 | 0.89 | cache fill is B-free | ✓ identical |
| hit_cache_respond_mixed (JIT-all) | 8.00 (B=1 miss) | 8.64 | batched JIT, latency-bound | ✓ +8% for ×4 rows — M2 claim confirmed |
| proxy_wait + draft_recv_cmd (IDLE) | 6.0 | 34.5 | — | draft slack GREW (see 1c) |
| top-level sum vs wall | — | 124.49 vs 124.28 | — | ✓ no unaccounted gaps / sync storms |

No draft label is grossly above its B×rows expectation. The suspected
hidden costs (per-seq mask rebuild at step 0, nested fan_out_list
numpy loops, sync storms) are all measured ≤ +0.3 ms and the wall is
fully accounted for by labels.

### 1b. Target labels (any-miss steps, n=1112 by target status)

| label | B=1 (hit_k1) | B=4 | verdict |
|---|---|---|---|
| graph_pre | 31.54 | 78.94 | ✓ physics: 40 verify rows vs 10; marginal 2.23 ms/row (see 1d), matches the ~2.2 ms/pos verify physics of tax_decomposition |
| graph_post | 12.14 | 30.11 | ✓ same |
| target_spec_wait | 2.68 | 3.00 all-hit / 10.84 any-miss | ✓ hit-step wait = B=1 baseline; miss stall structural (§2c) |
| verify_sample_accept | 3.64 | 2.40 | ✓ CPU hides behind longer GPU |
| proxy_compute_send | 1.48 | 0.50 | ✓ shrank (hidden) — mid-verify block is NOT a growing B-cost |
| exit_logits | 0.78 | 0.63 | ✓ flat |
| final_logits | 0.36 | 0.58 | ✓ tiny |
| top-level sum vs wall | — | 121.4 vs 122.6 | ✓ accounted |

### 1c. Which side binds at B=4: the TARGET

| | B=1 | B=4 |
|---|---|---|
| draft wall / idle / work | 52.1 / 6.0 / 46.1 | 122.1 / 34.5 / 87.6 |
| target wall (mixed / all-hit) | 52.3 | 124.3 / 119.2 |

Draft work grew ×1.90 while target verify grew ×2.5 — the draft's
slack WIDENED (6→34.5 ms). The 13 serial forwards never bind on hit
steps (spec_wait 3.0 ≈ B=1's 2.7); the draft touches the critical path
only through the JIT-response on any-miss steps (+7.8 ms, §2c). The
docs/duet/12 ranking that put "fewer/fatter draft forwards" first was
half-right for the wrong reason: fat7 wins mostly by narrowing the
VERIFY width (K1+1), not by saving draft forwards.

### 1d. vk_max width distribution and the padding tax

Step dispatch measured from graph_pre bimodality (cut 60 ms):

| dispatch | share of steps | graph_pre+post (ms) |
|---|---|---|
| k1 (vk_max=9, 40 rows) | 93.3% | 110.8 |
| k2 (all-short, 20 rows) | 5.8% | ~66 |
| ramp (partial batch) | 0.9% | — |

(B=1 for contrast: k1 59% / k2-or-miss 41%.) All-short probability
matches theory: per-seq long (P1-hit) share 0.553 → 0.447^4 ≈ 4%
predicted vs 5.8% measured (seq correlation). Marginal verify row cost
(110.8−66)/20 ≈ **2.23 ms/row**. In a k1-dispatch step
E[short seqs | ≥1 long] ≈ 1.65 × 5 padded rows ≈ 8.3 wasted rows →
**+18.4 ms × 93.3% ≈ 17.2 ms/step padding tax**. Cross-check: fat7's
width cut 40→32 rows moved T_verify −20.9 ms (§3) — the two estimates
bracket the tax at **17-21 ms/step**.

## 2. Causal decomposition of the champion-shape B=4 gap

Gap: 118.00 vs 150.31 = −21.5%; R = tok ratio 0.910 × step-time ratio
0.863. Time side, ΔT_target = 129.8 − 113.7 = **+16.1 ms** vs C:

| term | ms/step | how measured |
|---|---|---|
| vk_max-padded verify width | **+17.2** (17-21) | §1d row model; fat7 −20.9 confirmation |
| mid-verify DUET block (exit_logits + proxy_compute_send) | +1.1 | profile, mostly hidden |
| unpadded row diff (DUET 31.1 vs C 32 rows) | −2.2 | 2.23 ms/row |
| miss-stall difference (§2c) | −1.1 .. −4.7 | DUET 0.570×7.8 vs C 0.700×(8-13) |
| serial-forward count (13 vs 7) | ~0 direct | draft never binds on hit steps (§1c) |
| residuals (response-path glue at vk_max width, wire) | +1..+2 | profile |
| **sum** | **+12..+16** | **measured +16.1 ✓ closes** |

Token side: champion tok/step 3.63 (PROFILE=0) vs 3.91 (PROFILE=1,
same args/seed) — single-run noise at B=4 spans ±4%; C 3.99. The
−2..−9% token deficit equals DUET's B=1 deficit (m6: L_p2/miss-tok/P2
all B-invariant) — **not a B-effect**.

### 2c. The miss-stall amplification term (finding 5b's mechanism)

IS present, with measured magnitude:

| | B=1 | B=4 |
|---|---|---|
| DUET any-miss share (1−h^B) | 0.19 | 0.570 (measured 0.565 ✓) |
| C any-miss share | 0.26 | 0.700 |
| frequency advantage | 6 pts | **13 pts — amplification confirmed** |
| DUET stall per any-miss step | 7.4 (spec_wait Δ) | 7.8 (10.84−3.00) |
| C stall per any-miss step | ~12.8 (13.4 ms K7-JIT, scaled) | 8-13 (est., no C profile) |
| DUET net advantage | ~+1.8 ms/step | **+1.1 .. +4.7 ms/step** |

The term grows with B exactly as hypothesized — but at B≤4 it is
single-digit ms/step, an order below the padding tax it was supposed
to offset, and C's flat tok/step shows C amortizes its stalls. The
hypothesis' mechanism is real; its magnitude cannot carry a win alone.

## 3. Exp2 — B>1-shape retune probes (B=4)

Both cells: champion base args, jit-short on, PROFILE=0.
fat7 = K1=7 K2=4 (k=11), uniform dfo=2 ([2]×8 = 16 rows/seq phase-1,
11 serial forwards, verify 8 pos/seq = 32 rows). fat5 = K1=5 K2=4
(k=9), uniform dfo=3 ([3]×6 = 18 rows/seq, 9 forwards, verify 24
rows; needs --f 4 so dfo<f, pfo stays 1 → phase-2 budget 6).

| metric | champion b4 | prof_b4 (=champion+prof) | fat7 | fat5 | C b4 |
|---|---|---|---|---|---|
| Decode TPS | 118.00 | 126.69 | 144.72 | **155.12** | 150.31 |
| vs C | −21.5% | −15.7% | −3.7% | **+3.2%** | — |
| Tok/step | 3.63 | 3.91 | 3.71 | 3.41 | 3.99 |
| T_target (ms) | 129.82 | 131.74 | 108.95 | **95.35** | 113.71 |
| T_verify (ms) | 112.88 | 113.85 | 91.97 | **79.56** | 91.42 |
| T_draft (ms) | 103.70 | 105.22 | 86.99 | 75.54 | 75.28 |
| Cache hit | 0.80 | 0.81 | 0.82 | **0.84** | 0.74 |
| P1 hit / L_p1 | 0.529 / 3.50 | 0.553 / 3.84 | 0.591 / 3.42 | 0.660 / 2.83 | — |
| P2 hit / L_p2 | 0.274 / 1.63 | 0.261 / 1.83 | 0.230 / 1.69 | 0.175 / 1.67 | — |
| step t = B·tok/TPS (ms) | 123.1 | 123.5 | 102.5 | **87.9** | 106.2 |

fat7 findings: verify width parity with C (32 rows) → T_verify parity
(91.97 vs 91.42). Step time now FASTER than C (102.5 vs 106.2 —
DUET's remaining verify advantage + cheaper stalls). K1 9→7 cost
almost no tokens (L_p1 3.42 vs 3.50 — the champion's fo=1 tail
positions 8-9 carried ~0.1 tok) while P1 hit rate ROSE (0.591). The
entire remaining −3.7% is token-side (0.930 ratio) — DUET's known B=1
deficit (L_p2 1.69 vs breakeven ~2.6, docs/duet/12 finding 5a).

fat5 findings: the fat trade WINS at B=4. Verify 24 rows (25% below
C's 32) → T_verify 79.56 (−13% vs C); T_draft 75.54 ≈ C's 75.28 with
9 serial forwards; hit rate 0.84 (any-miss burden 1−0.84^4 = 0.50 vs
C 0.70). Tokens drop as K1=5 caps chains (L_p1 2.83, tok/step 3.41 =
0.855 of C) but the step-time ratio (t_C/t_D = 106.2/87.9 = 1.208)
more than pays for it: R = 0.855 × 1.208 = 1.033 → **+3.2% measured
(155.12 vs 150.31)**. Note fat5's Marlin geometry: phase-1 B×18 = 72
rows (4.5 tiles — crosses the tile boundary and STILL wins, because
at B=4 the tile cliff is amortized over 4 seqs) and its per-step
token cost is the worst of the three DUET cells — the win is pure
step-time shape.

## 4. Honest final statement

1. **No bug.** All 26 profile labels match their B×rows structural
   models; walls are label-accounted on both procs; hit-side cache
   fill and batched JIT behave exactly as designed (M2/M3/M6 all
   hold at B=4 under profile).
2. **The champion SHAPE was a B=1 artifact.** K1=9 deep-narrow was
   tuned to the B=1 16-row tile cliff + 41% short-dispatch mix. At
   B=4, 93% of steps pay K1-width verify for every seq (the v1
   uniform-vk_max design cost) — 17-21 ms/step. The fix is not code,
   it is shape: **use fat5 (K1=5 K2=4 dfo=3 uniform, k=9, --f 4) at
   B=4** — 155.12 tok/s, **+3.2% over SD-best C**, the first
   measured B>1 DUET win (docs/duet/12 finding 5b: partially
   CONFIRMED at v1, via shape retune, single run). fat7 (K1=7,
   144.72, −3.7%) is the token-conservative alternative.
3. What remains as levers at B=4: DUET's token-side deficit
   (tok/step 0.86-0.93 of C, B-invariant, the L_p2≈1.7 off-policy
   continuation quality — docs/duet/12 finding 5a), per-seq verify
   dispatch to reclaim the residual padding inside mixed batches,
   and a proper per-B shape sweep (fat5 was the FIRST fat-shape
   guess; K1∈{4,5,6} × dfo × pfo at B∈{2,4,8} is unexplored, and
   fat5 at B∈{1,2} is unmeasured — the B=1 champion stays E9K24_jit).

Caveats: single run per cell (champion tok/step spans 3.63-3.91
across two identical-args runs — token-side comparisons at ±4%);
C-side stall estimated without a C profile; fat5 uses --f 4 (wider
miss JIT than the other cells); PROFILE=1 adds CUDA-event overhead yet
prof_b4 ran faster than the PROFILE=0 champion cell — the delta is
token-draw noise, not profiling speedup.

Repro: `run_prof_b4.sh`, `run_retune.sh`, `analyze_prof.py` (this
directory); raw profile JSONs in `prof_b4/` (subject to the standing
prune policy, docs/duet/12).
