# 09 — Beat-SD-best campaign plan

**Date**: 2026-07-03
**Goal**: DUET decode TPS must beat async SD's best operating point
(C: k=7 f=6) with non-overlapping 3-rep bands, measured on the same GPU
set. Prior gap: A 81.24±0.67 vs C 82.72±0.41 → −1.48 tok/s (−1.8%).

**GPU note**: GPUs 0-1 are held by another user (fnsl1026 vLLM) as of
2026-07-03. All campaign runs use `CUDA_VISIBLE_DEVICES=2,3,5,6,7`
(target TP4 on 2,3,5,6; draft on 7). Absolute TPS is not comparable to
the 0-4 runs; every claim in this campaign is made against baselines
re-measured on the SAME set.

## Where the gap lives (established)

1. Widening (dfo/pfo) — REJECTED: substitution effect (wide_sweep).
2. Deepening K1 — token-correct (K1=9 → 4.27 tok/step > C's 4.15),
   time-rejected: target step grows **+3.4 ms per K1 position** against
   a ~1.26 ms verify-CG projection (deep_p1_test). Break-even: the tax
   must fall to ~0.6 ms/pos for K1=9 to beat C, ≤1.83 for K1=8 to pay.
3. L_p2 ≈ 2.0 is the fundamental algorithmic residual (off-policy
   continuation ≈ cached JIT chain) — not attackable in a systems paper.

Therefore the campaign is a **systems attack on the per-position tax**,
plus deterministic small harvests.

## Workstreams

### WS1 — Tax decomposition (diagnostic; experiments/proxy_async_overlap/tax_decomposition)

3 profile points K1 ∈ {7,8,9}, PROFILE_DUET=1, ns=20. Per-label
per-status means, fit slope per label; 3 points separate linear growth
from CG-bucket step jumps.

- **Q1 (target)**: split +3.4/pos into verify CG compute vs spec_wait
  growth. spec_wait growth = pipeline echo of the draft's +4.0/pos
  (draft slack was only 5.1 ms at K1=7 — draft +8 ms at K1=9 must
  surface somewhere).
- **Q2 (coupling)**: does proxy arrival (t_exit) shift later with K1?
  Exit-layer output exists only after verify reaches layer 56 of a
  (K+1)-token forward → deeper K1 delays phase2 start ≈ +0.9 ms/pos
  (0.7 × verify slope). If confirmed, deepening self-throttles: it eats
  the very idle it was meant to fill.
- **Q3 (draft)**: attribute +4.0/pos beyond phase1's +2.5 (glue width
  K1+1, logits_q unpack, merge/respond, JIT depth on miss).

### WS2 — Deterministic harvests (independent)

- **JIT depth pin**: split-mode JIT is already pinned to K_max=K1 (not
  K_long). Remaining candidate: pin BELOW K1 (e.g. 5). Miss steps pay
  ~13 ms JIT ≈ K1 unbatched draft forwards; depth 5 saves ~2 fwds
  (~3.7 ms) × 18.8% miss share ≈ +0.7 ms/step, minus a small L_miss
  truncation loss (L_miss 2.18 → P(L≥5) small). Env-gated
  `SSD_DUET_JIT_K` for A/B.
- Re-test `SSD_ASYNC_PROXY_SEND` (+`SSD_PROXY_STREAM`) as a ride-along
  after WS3 lands (gain was UNCONFIRMED pre-cleanup; interactions
  possible).

### WS3 — Targeted fix (conditional on WS1's attribution)

| if the dominant slope is | attack |
|---|---|
| target spec_wait (draft echo) | draft-side +4/pos: phase1 batch shape, glue width, unpack cost |
| verify CG compute (step jumps) | CG bucket / Marlin tile boundary audit at B×(K+1) rows |
| proxy compute / exit gather | cap proxy positions to K1+1 (phase2 rows can't be chosen_pos > K1) or move to proxy_stream |
| verify_sample_accept | vectorize, remove per-position syncs |

Success = per-position tax ≤ ~1.8 ms (K1=8 viable) or ≤ ~0.6 (K1=9 wins).

### WS4 — Verdict rematch

Best DUET config (A+fixes vs K1=8/9+fixes, whichever scans best) vs C,
3 reps each, interleaved per cycle on GPUs 2,3,5,6,7. Pre-registered
rule: DUET mean > C mean AND no band overlap → campaign success.

## Risk register

- Peer vLLM job on GPUs 0-1 contends for PCIe/host resources → slope
  analysis is diff-based (robust), but final rematch reps are
  interleaved to spread any drift.
- If WS1 shows the tax is irreducible verify compute (Marlin small-batch
  physics), fallback claims: (a) A + JIT pin alone vs C (gap ~−1% →
  needs luck), (b) miss-rate/p99 latency-jitter framing, (c) DUET wins
  the draft-compute-constrained regime (bigger draft, B>1) where SD
  cannot afford f=6.
