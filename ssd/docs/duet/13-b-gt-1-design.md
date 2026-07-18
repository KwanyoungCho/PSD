# 13 — B>1 support for DUET split-K1/K2: design + staged plan

**Date**: 2026-07-18. Motivation: docs/duet/12 finding 5(b) — B>1 is
DUET's structural home turf (draft forwards are latency-bound, so seqs
share them nearly free; DUET needs 26 rows/seq vs SD-best's 48, so it
batches deeper before hitting compute walls). Full B=1-assumption audit:
see the survey table below and config.py L308 (the stale "Policy A"
comment — the real blocker is the single-seq Policy B + single-bucket
dispatch).

## Core design: uniform-width batch, per-seq masks (v1)

The audit's central fact: non-DUET SSD already batches because all seqs
share ONE layout. DUET's heterogeneity is three things — per-seq
valid_k ∈ {K1,K2}, per-seq Policy-B fan-out, per-seq proxy wire. v1
restores a shared batch SHAPE and pushes all per-seq variation into
masks/index tensors (replay-time buffers), so **no new CUDA-graph
families are needed** — only the existing bs-bucket axis.

1. **valid_k mixing → dispatch on `vk_max = max(valid_k)` over the
   batch.** Glue width, phase-1 layout (long/short), verify bucket, and
   the TP-broadcast scalar all use vk_max (stays a single scalar for TP
   sync). Short seqs are padded to vk_max; their REAL width rides in
   `valid_k [B]` and is enforced at exactly one point per consumer:
   - verify(): clamp `accept_until = min(accept_until, valid_k_i)`
     (one vectorized line) — padded positions can never be accepted.
   - glue/tree masks: already built per-seq (cache_hits_list loop) —
     tril width per seq follows valid_k_i.
   - Cost: an all-short batch pays K1-width verify unless all seqs are
     short (then dispatch k2 bucket — cheap max check). Acceptable v1
     loss; P(all-short) shrinks with B anyway.
2. **Mixed hit/miss (audit #5)**: reverse the JIT/cache priority — run
   the batched JIT for ALL rows (latency-bound: extra rows ~free), then
   overwrite hit rows from the cache. Per-seq valid_k = hit ? row_vk :
   K2 (jit-short). Kills the JIT-all-clobbers-hits bug that would cost
   ~55% of steps at B=4.
3. **Policy B batched (audit #2)**: `_compute_and_send_proxy` grows a B
   axis end-to-end — accept_probs/h/cumprod [B,K], P_iv [B,K+1,top_k],
   per-seq topk(wire_N) → chosen [B, wire_N]. Wire/irecv/ring sized
   2·B·wire_N. `duet_policy.policy_b_from_candidates` gets a batch dim
   (both consumers keep working). Padded positions of short seqs get
   h=0 (α̂ padded to 0) so their P_iv mass is 0 — chosen never lands
   beyond vk_i.
4. **Phase-2 dynamics (audit #3, #4)**: the key invariant that saves
   us: per-seq budget sum is CONSTANT (= duet_proxy_total_budget), so
   the phase-2 batch shape is uniform [B × MQ_p2] × K2 forwards — CG
   shape untouched. Only the DISTRIBUTION varies per seq:
   - `_select_proxy_sourced_tokens_unified`: vectorize dedup/cumsum/
     argsort over B → proxy_forked [B, MQ_p2], fan_out [B, vk_max+1]
     (rows for pos > vk_i are 0 by construction of chosen).
   - `_update_phase2_layout_inplace`: fan_out_t becomes [B, P],
     fan_idx per-seq concatenated [B·MQ_p2]; position_count = vk_max+1
     uniform. The ONE .tolist() sync becomes a [B, P] tolist (same
     single sync).
   - mask builder: per-b loop already exists — feed per-seq
     fan_out_list (list of lists) instead of the shared list.
   - rope_positions: already per-row via fan_idx — just per-seq fan_idx.
5. **Speculator/verifier plumbing**: replace the four uniform-width
   collapses (speculator L145-167 slice, verifier L110-114 unique
   assert, L137 scalar lookahead, L186 view) with vk_max + the accept
   clamp. logits_p view uses vk_max+1 uniformly (padded rows wasted —
   part of the v1 padding cost).
6. **Out of scope v1** (assert B==1 where entered): exit_topm_gather /
   exit_replica / proxy_on_draft gates (all off-champion, measured
   neutral), EAGLE path. duet_exit_topm's y_tok slice crosses seq
   boundaries at B>1 — guarded, not fixed.
7. **Config**: lift the max_num_seqs==1 assert to ≤8 for DUET; fix the
   stale Policy-A comment; wire_N stays per-seq (B factor applied at
   send/recv sites).

## Why this should be fast (systems view)

- Draft: phase-1 B×16 rows/fwd, phase-2 B×10 rows/fwd — at B=2 exactly
  2 Marlin tiles (32 rows, full utilization): per-seq draft cost ≈
  halves. JIT batches across miss rows. Glue batches (varlen already).
- Target: verify B×(vk_max+1) rows — the m-dim grows into better GEMM
  utilization on the 70B side too; verify() and metrics are already
  vectorized.
- No new CG families ⇒ capture time and memory grow only along the
  existing bs-bucket axis ({1,2,4} initially).
- The scheduler/wire/step layers are already B-generic (audit §b).

## Staged plan (one commit + smoke per stage)

| stage | content | validation |
|---|---|---|
| M1 | Verifier: batched Policy B + wire 2·B·wire_N; draft irecv/unpack B axis; speculator vk_max slicing; verifier uniform-assert → vk_max + accept clamp | B=1 regression smoke (must be byte-equivalent path) |
| M2 | Draft: vk_max glue/layout dispatch, per-seq valid_k plumb, mixed-JIT fix (JIT-all-then-cache-overwrite) | B=1 smoke + unit test of clamp/pad math |
| M3 | Batched selector + per-seq phase-2 fan-out (fan_out_t [B,P], per-seq fan_idx/masks/rope) | CPU unit test vs B=1 selector reference |
| M4 | Config gate lift (≤8); B=2 end-to-end smoke; correctness check (B=2 tok/step ≈ B=1 within sampling noise; no hit-rate collapse) | ns=8 B=2 vs 2× B=1 |
| M5 | Perf: B ∈ {1,2,4} champion vs SD-best B-sweep, same GPU set, interleaved | the regime-win measurement (docs/duet/12 finding 5b) |

Risks: (a) per-seq fan_out zero-rows at padded positions must keep
repeat_interleave/mask math consistent (M3 unit test); (b) TP bucket
sync relies on vk_max being derived identically on rank 0 and draft —
it travels on the existing valid_k wire, computed once draft-side;
(c) scheduler admission at B>1 multiplies lookahead reservations —
watch preemption (the B=0 guard already exists).

## M1 — implemented (2026-07-19)

Landed per design §1/§5: batched Policy B in `_compute_and_send_proxy`
(h/cumprod/P_iv with a B axis, per-seq topk(wire_N) → chosen [B,wire_N],
wire flattened to 2·B·wire_N; ring sized with max_num_seqs), draft
`_irecv_duet_proxy`/`_unpack_duet_proxy` B-axis ([B,wire_N] views;
selector consumes seq 0 until M3), speculator seq-0 `_vk_scalar` →
vk_max, verifier `torch.unique` assert → `valid_k.max()` (sync swap,
not an addition), `verify(valid_k=...)` per-seq accept clamp
(`accept_until = min(accept_until, valid_k)`), B==1 guards on the
off-champion gates (topm/replica/proxy-on-draft, §6).

Validation: unit tests ssd/tests/test_b_gt1_m1.py — 9/9 OK (batched
math ≡ single-seq reference at B=1/2/3; clamp semantics). B=1 GPU
regression smoke (champion config, ns=4/out=128): 71.50 tok/s,
L_p1 3.46, cache 0.80, no errors — within the established ns=4 noise
band (58.8–75.8 across prior smokes). B=1 wire length 2·1·wire_N is
unchanged; all reshapes are no-ops at B=1.
