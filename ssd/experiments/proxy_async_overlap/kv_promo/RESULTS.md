# KV promotion (SwiftSpec-style glue removal) — correct, but TPS-neutral at B=1

**Date**: 2026-07-04. Gate `SSD_DUET_KV_PROMO=1` (default OFF).
Design docs/duet/11; motivation docs/duet/10 item 1.

## Correctness — VERIFIED

Parity mode (`SSD_DUET_KV_PROMO_PARITY=1`: run promo, snapshot promoted
glue KV/logits, run legacy glue, diff) after the root-cause fix:
- kv_maxdiff/pos ~0.06-0.17 (fp16 recompute noise — glue forward vs tree
  forward differ at fp16), for BOTH draft rows and proxy rows, ALL depths.
- tip logit_maxdiff ~0.1-0.25 (was 7-11 before the fix — the tip forward
  now reads correct gathered context).

Root cause found (docs/duet/11): the draft's scratch/lookahead KV blocks
are re-mapped to fresh physical blocks each step, so old scratch
positions resolve to the WRONG physical block under the promo step's
dbt. Fix: address the gather source through the BUILD-time dbt (stashed
in `_promo_meta`); the old physical blocks still hold the build KV (B=1,
no draft write between build and promo). Commit 4ea74e8.

## A/B (ns=20, out=256, same seed, champion E9K24_jit)

| metric | promo OFF | promo ON | Δ |
|---|---:|---:|---:|
| L_p1 | 3.70 | 3.68 | −0.02 (equal) |
| tok/step | 3.77 | 3.79 | +0.02 |
| cache hit | 0.80 | 0.81 | +0.01 |
| T_target (ms) | 52.00 | 52.00 | 0 (draft-side change) |
| **T_draft (ms)** | 44.63 | **45.25** | **+0.62** |
| TPS | 76.06 | 76.49 | +0.43 (noise) |

## Verdict — mechanism works, but no draft-time win here

Acceptance is preserved (L_p1/hit equal → promotion is functionally
correct). But T_draft went UP +0.62 ms. Why: glue processes vk+1=10
tokens in ONE latency-bound forward (~1.78 ms); promo replaces it with a
1-token tip forward (still ~1.2 ms — latency-bound) + a KV gather across
all layers (~1 ms). Same forward count, plus gather overhead → net
draft-side loss. This is the campaign's recurring lesson: latency-bound
forwards mean "less work per forward" saves nothing; only FEWER forwards
help. Glue = 1 forward; promo = 1 forward + gather.

At the B=1 champion the draft has slack and isn't binding, so T_draft
+0.62 is absorbed and TPS is unchanged (76.06 vs 76.49 within noise).

## To make it a real win

Eliminate the tip forward too (glue → 0 extra forwards, gather only) by
folding position-vk's KV+logit into phase-1's forward 1 (mask surgery +
a position-K sub-row — docs/duet/11 "later optimization"). Then promo
saves the full ~1.78 ms glue forward. Even then, at B=1 champion this is
draft slack (no TPS) — it pays off only (a) bundled with wide-early/
exit-pull overlap that consumes the freed draft budget, or (b) in
draft-bound regimes (bigger draft, B>1), where draft time is the period.

## Status

Implemented, correct, committed, gated OFF. Kept as an enabler; not on
the champion path. The frontier lever remains target-side (fused
GEMM+AllReduce kernels, docs/duet/10 item 2).

## PROFILE confirmation (ns=20, PROFILE_DUET=1, per hit_k1 step, ms)

| draft label | OFF (legacy glue) | ON (promo) |
|---|---:|---:|
| `glue` span (total) | 5.44 | 5.41 |
| └ draft_glue_replay (the forward) | 3.45 | — (none) |
| phase1_replay | 44.3 | 45.0 |
| phase2_replay | 19.6 | 20.3 |

The `glue` span is EQUAL OFF≈ON (5.44 vs 5.41): the legacy glue forward
(3.45 ms) + prep is replaced by promo's gather + tip forward at the SAME
cost — a wash. Confirms the mechanism runs but saves no draft time,
because the 10-token glue forward is already cheap (GPU batch is free /
latency-bound), so removing it and paying a 1-token tip forward + KV
gather nets to zero. Absolute values run high vs the cold-path champion
profile (PROFILE=1 sync overhead; hit_k1 share differed 0.62/0.34 across
the two runs — warmup mix), but the same-condition OFF/ON glue-span
equality is the robust result.

**Final status: KV-promo is correct and complete, gated OFF, not on the
champion path. A wash at B=1 (draft not binding + batch-free glue). Kept
as a draft-bound-regime enabler.** SwiftSpec's remaining kernels are
Hopper sm_90 silicon (wgmma/TMA/clusters, NVLink IPC) — unusable on the
RTX 3090 (sm_86) without a ground-up rewrite that loses the point.
