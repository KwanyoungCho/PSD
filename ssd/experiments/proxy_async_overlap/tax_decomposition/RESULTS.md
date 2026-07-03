# Tax decomposition — the 3.4 ms/pos is a REGIME CHANGE, not a smooth cost

**Date**: 2026-07-03
**Setup**: K1 ∈ {7,8,9}, K2=5 exit=56 dfo=2 pfo=1 (uniform), ns=20,
SSD_PROFILE_DUET=1, GPUs 2,3,5,6 (target TP4) + 7 (draft) — GPUs 0-1 held
by another user. Full tables: `slopes_3pt.md` (analyze.py output).

Headline TPS (PROFILE=1, this GPU set): K1=7 77.01, K1=8 78.70, K1=9 73.73.
Note K1=8 > K1=7 here — the deep_p1 ordering on GPUs 0-4/PROFILE=0 does
not transfer 1:1; all campaign verdicts are re-measured on this set.

## Attribution (hit_k1 steps, ms per +1 K1 position)

| component | slope | linear? | note |
|---|---:|---|---|
| target graph_pre+post (70B verify CG) | +2.22 | ✓ (resid +0.3) | physics: attention+lm_head+allreduce per verify position |
| target spec_wait (draft echo) | +2.29 | accelerating | draft overrun leaks into target wait |
| target proxy_compute_send | −0.36 | — | shrinks as draft waits earlier |
| target verify_sample_accept | −0.25 | — | CPU hides behind longer GPU |
| **target step wall** | **+3.73** | | 49.95 → 53.97 → 57.40 |
| draft phase1_replay | +7.52 | **NO** (resid +3.7) | **+11.2 at 7→8 (tile cliff), +3.8 at 8→9** |
| draft proxy_wait (idle) | −4.04 | NO | 8.13 → 0.15 → 0.05: idle GONE at K1≥8 |
| draft phase2_replay | +1.22 | NO | +2.4 at 7→8 then flat (clock/contention echo) |
| draft recv_cmd (idle) | −1.43 | | second idle pool also drained |

## The mechanism

1. Phase-1 MQ_LEN = dfo×(K1+1) = 16 rows at K1=7 → 18 at K1=8 crosses the
   draft's Marlin m-tile: per-forward 2.52 → 3.61 ms (+44%) — ALL positions
   pay, not just the new one (+11.2 ms at the crossing).
2. glue+phase1 (22.4 ms) blows past proxy arrival (~29 ms): draft idle
   vanishes, phase2 starts late, the draft finishes after the target wants
   its next speculation → spec_wait echo +2.3/pos on the target wall.
3. On top: the target's own verify CG costs +2.2/pos (smooth — no tile
   effect on the 70B side), and miss steps pay +1.8/pos more JIT
   (hit_cache_respond_miss 13.4 → 15.2 → 17.0).

## Fat constants exposed at K1=7 (per step, hit_k1)

| item | ms | shared with SD baseline? |
|---|---:|---|
| verify_sample_accept (CPU sync/dispatch chain) | 3.60 | YES — does not close the gap |
| proxy_compute_send (Policy B on verify path) | 1.43 | **NO — DUET-only** |
| exit_logits (lm_head+gather mid-verify) | 0.66 | NO (needed for proxy) |
| draft prep total (glue+p1+p2 CPU prep) | ~5.3 | draft-side, hidden while idle exists |
| miss JIT (K1-deep, unbatched) | 13.4 | depth is DUET-controllable |

## What this buys (implemented, gated OFF, in gate_scan)

1. **Deep-narrow phase1** (`--duet_split_phase1_fan_out_list 2,...,1,...`,
   sum=16): K1=8/9 depth at 16 rows — no tile cliff, idle preserved, no
   echo. Tail positions at fo=1 are near-free (h-mass concentrates early).
2. **SSD_DUET_PROXY_ON_DRAFT**: Policy B moved into the draft idle window;
   target sheds its DUET-only ~1.4 ms/step (and the −0.36 slope artifact).
3. **SSD_DUET_JIT_SHORT**: miss JIT K1→K2 deep (−1.9 ms/fwd × (K1−K2)) AND
   the next verify shrinks K1+1→K2+1 positions (×2.2 ms each) — at K1=9
   that is ~−7.6 − 8.8 ms on miss steps (19% share).

Predicted stack at K1=9 deep-narrow + both gates: tokens 4.27/4.02 with a
target wall near baseline → decisively above C. gate_scan verifies.
