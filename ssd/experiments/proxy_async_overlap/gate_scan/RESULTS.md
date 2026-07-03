# Gate scan — JIT-short is the big lever; deep-narrow delivers depth tokens; E9_jit beats C

**Date**: 2026-07-03
**Setup**: 70B AWQ TP=4 (GPU 2,3,5,6) + TinyLlama AWQ (GPU 7), ns=50
in=512 out=512 --all, seed=42 temp=0.7, PROFILE=0. GPUs 0-1 held by
another user throughout — absolute numbers are NOT comparable to the
GPU 0-4 series; C is re-measured here as the bar.

## Main scan (1 run per cell) + repeats

| cell | gates | TPS | tok/step | T_target | T_draft |
|---|---|---:|---:|---:|---:|
| A_base | — | 77.30 | 3.96 | 52.53 | 47.11 |
| **C_sd (bar)** | — | **81.89** | 4.10 | 51.44 | 39.70 |
| A_jit | J | 81.15 / **82.35** (r2) | 3.97 | 50.17 | 45.03 |
| A_pod | P | 79.03 | 3.99 | 51.79 | 46.68 |
| A_jit_pod | J+P | 79.47 / 81.13 (r2) | 4.00 | 51.66 | 46.17 |
| E8_deep16 | J+P | 79.68 | 4.10 | 52.75 | 48.77 |
| E9_deep16 | J+P | 79.90 | 4.19 | 53.79 | 49.64 |
| **E9_jit** | J | **83.25** | **4.31** | 53.14 | 49.31 |

J = SSD_DUET_JIT_SHORT, P = SSD_DUET_PROXY_ON_DRAFT.
E8/E9 = deep-narrow phase1 fan_out_list (sum=16, Marlin-tile-safe):
E8 `2×7,1×2` (K1=8), E9 `2×6,1×4` (K1=9).

## Findings

1. **JIT-short: +3.85 (A 77.30→81.15), pure time win.** T_target −2.36 ms
   with hit rates and tok/step unchanged — the designed mechanism (K2-deep
   JIT on miss + miss's next verify shrinks K1+1→K2+1 positions) and
   nothing else. Single biggest lever found in the campaign.
2. **Deep-narrow delivers the depth tokens with no tile cliff and no
   substitution.** L_p1 3.60→3.91→4.25 (A→E8→E9_jit) tracks the
   truncated-geometric model; per-forward stays ~2.5 ms (16 rows);
   p2_hit *rises* (0.227→0.241) instead of being cannibalized (contrast:
   dfo-widening rejected in wide_sweep).
3. **proxy-on-draft works alone (+1.73) but adds nothing over JIT-short.**
   First combo run (79.47) suggested a −1.5 ms drag; the repeat (81.13)
   ≈ A_jit r1 — the "drag" was mostly run noise. Combo ≈ J alone, so the
   champion config omits P (simpler = better). P remains valuable as a
   standalone result: it removes the DUET-only proxy cost from the
   target verify path (verifier softmax/topk/pack → draft idle window),
   CPU-equivalence-tested at 100% overlap.
4. **Run noise on this GPU set is σ ≈ 0.6-1.7 tok/s** (A_jit 81.15/82.35;
   A_jit_pod 79.47/81.13) — larger than the GPU 0-4 series (σ 0.4-1.7),
   plausibly the neighbor's vLLM load varying. Hence the 3-rep
   interleaved final_rematch for the verdict.

## Champion

**E9_jit** = split-K1/K2, K1=9 K2=5, exit=56, pfo=1,
phase1 fan_out_list `2,2,2,2,2,2,1,1,1,1` (sum 16), SSD_DUET_JIT_SHORT=1:
tok/step 4.31 (+5.1% over C) at T_target 53.14 (+3.3%) → **83.25
(+1.36 / +1.7% over C's single run)**. 3-rep verdict: final_rematch/.
