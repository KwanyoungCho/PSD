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

## M2 — implemented (2026-07-18)

Landed per design §1/§2 (draft side, `ssd/engine/draft_runner.py` +
`utils/async_helpers/async_spec_helpers.py`):

1. **vk_max dispatch** (§1): `hit_cache_and_respond`'s dispatch scalar
   `_vk_scalar` switches from the seq-0 capture `valid_k[0].item()` to
   `valid_k.max().item()` — a sync SWAP (still exactly one GPU→CPU sync
   per step). The scalar remains the ONLY batch-level width: it drives
   `make_glue_decode_input_ids` slicing, `prepare_glue_decode_ctxt` /
   `_glue_decode` bucket dispatch, and the phase-1 long/short layout
   pick in `_build_tree_batch_split_k1k2` — all with unchanged
   signatures. Per-seq `valid_k [B]` rides the response wire and
   `partial_tree_decode_args` untouched. Rows with vk_i < vk_max feed
   filler to the vk_max-wide glue beyond vk_i (cache-padding zeros for
   short hits, random-init in-vocab tokens for JIT-short misses) — safe
   because phase-1 fork selection slices positions per layout and the
   M1 verify clamp (`accept_until = min(accept_until, vk_i)`) makes
   forks past a short row's vk_i unreachable cache rows (the v1 padding
   cost, not a correctness issue). Mixed K1/K2 batches dispatch the
   long bucket; only an all-short batch takes K2 (cheap max check).
2. **Mixed hit/miss fix** (§2, the JIT-all-clobbers-hits bug): the fill
   logic in `hit_cache_and_respond` is restructured from
   "all-hit → cache fill ELSE JIT-all" to "JIT-all on ANY miss
   (unchanged batched call, latency-bound so extra rows are ~free),
   THEN overwrite hit rows from the cache". Hit rows keep their cached
   tokens/logits/valid_k/phase_source; miss rows keep JIT output with
   the JIT default valid_k (K2 under `SSD_DUET_JIT_SHORT`, else K_max)
   and phase_source 0. The `.any()/.all()` __bool__ syncs replace the
   identical syncs the old branch condition already paid — no new
   hot-path GPU sync.

**Why the mixed fix matters at B>1** (the user's hypothesis, docs/duet/12
finding 5b): base async-SD stalls the WHOLE batch on any single miss —
P(step degraded) = 1 − hit_rate^B, so at B=4 with hit≈0.80 that is ~59%
of steps. Pre-M2 DUET had exactly that failure mode: one miss threw away
every hit row's cached tree (~55% of steps at B=4 would clobber hits).
Post-M2 a miss costs only the missing row its cache benefit; hit rows
still verify their cached K1/K2 trees. DUET's higher hit rate (0.80 vs
SD-best 0.76, and cheaper misses via JIT-short) therefore COMPOUNDS with
B instead of being destroyed by it — this fix is what lets DUET realize
its structural B>1 advantage.

B=1 identity: at B=1 a batch is all-hit or all-miss, so the JIT branch
and cache fill run exactly as pre-M2 (all-hit → fill only, all-miss →
JIT only, no overwrite), and `max(valid_k) == valid_k[0]`.

Audit leftovers (§4): `_construct_tree_decode_args` is non-DUET-only
(left); `_build_tree_batch_split_k1k2` has no remaining B=1-scalar
indexing in M1/M2 scope (`_step_valid_k` = vk_max by design; the
`duet_proxy[...][0]` seq-0 collapse and the selector's `assert B == 1`
are M3 scope; `_policy_b_from_raw_proxy`'s `out_logits[0]` sits behind
the B==1-guarded off-champion raw-proxy gate, §6).

Validation: unit tests ssd/tests/test_b_gt1_m2.py — 8/8 OK (the REAL
`hit_cache_and_respond` on CPU via a stub DraftRunner: B=3 hit/miss/hit
keeps cached tokens/valid_k on hit rows and JIT output + K2 on the miss
row; JIT-long default; all-hit skips JIT; all-miss/empty-cache skip the
overwrite; all-short dispatches vk_max=K2; B=1 hit/miss identity).
M1 tests still 9/9 OK. B=1 GPU regression smoke (champion config,
ns=4/out=128): 71.35 tok/s, L_p1 3.49, cache 0.83, zero Tracebacks —
matches M1's 71.50/3.46/0.80 within noise (ns=4 band 58.8–75.8). Log:
experiments/proxy_async_overlap/b_gt1/m2_smoke/run.log.

## M3 — implemented (2026-07-18)

Landed per design §4 (batched selector + per-seq phase-2 fan-out;
`ssd/engine/draft_runner.py` + `engine/helpers/cudagraph_helpers.py` +
`engine/helpers/tree_layout.py`):

1. **Batched selector**: `_select_proxy_sourced_tokens_unified` drops its
   `assert B==1` and vectorizes over B — dedup via per-seq advanced
   indexing `draft_forked[b_idx, chosen_pos]` ([B,N,max_fo]; the
   [P,max_fo] mask broadcasts, shared across seqs since Phase 1
   fan_out_list is per-seq-uniform), budget cumsum along dim 1 with
   `take = valid & (rank <= total_budget)` per seq, boolean-index +
   `view(B, total_budget)` (row-major order + the exactly-total_budget
   invariant recover per-seq groups — same boolean-index op as pre-M3,
   no new sync), `scatter_add_` dim 1 → fan_out [B, K_rank+1] (each row
   sums to total_budget), per-seq stable argsort + gather for the
   pos-grouped result [B, MQ_p2]. The Fix-③ underfill guard becomes
   per-seq (`(take.sum(1) == total_budget).all()`, still `__debug__`-only).
   Call site: M1's seq-0 collapse removed; raw-proxy mode (B==1-guarded,
   §6) lifts its 1-D tensors to [1, wire_N].
2. **Per-seq layout**: `_update_phase2_layout_inplace` accepts
   fan_out [B, P]; `fan_idx = arange(P).repeat(B).repeat_interleave(
   fan_out.reshape(-1))` — the per-seq repeat_interleaves concatenated
   [B·MQ_p2] (bit-identical at B=1); `fan_out_list` becomes a list of
   per-seq lists via the ONE pre-existing `.tolist()` sync (now B×(≤9)
   elements); position_count = K_rank+1 uniform. New TreeLayout flag
   `fan_idx_per_seq` (default False, only the runtime-mutated split_k2
   layout sets it) tells consumers the fan_idx already spans all seqs:
   `_build_tree_decode_args_for_layout` (j_idx/rope_positions) and
   `_merge_and_populate_cache` (proxy_k cache keys) use it directly
   instead of re-cat-ing per seq. metadata F stays an int (seq 0's first
   entry — same value as pre-M3 at B=1).
3. **Per-seq masks**: `run_fi_tree_decode_cudagraph`'s glue-mask build
   detects the nested fan_out_list and builds one `np.repeat(_tril,
   fol_b)` block per seq; the per-b mask loop indexes `glue[b]` (padded
   CG-bucket rows reuse the last real seq's block — outputs discarded via
   slot_map -1). The `_cached_fol` cache keys on the FULL per-seq
   structure, so any distribution change forces a rebuild (the norm at
   B>1); at B=1 a repeated distribution still hits the cache as pre-M3.
   Flat lists (split_k1/full/non-DUET) keep the shared-glue path
   untouched. Phase 1 stays untouched (§4 — its per-seq-uniform list
   already batches).

B=1 identity: every batched op degenerates to the pre-M3 single-seq op
at B=1 (indexing/cumsum/scatter/argsort along the new dim over one row;
fan_idx values identical; nested `[[...]]` glue build produces the same
numpy block).

Validation: unit tests ssd/tests/test_b_gt1_m3.py — 7/7 OK (batched
selector vs a per-seq loop of the ORIGINAL pre-M3 selector copied into
the test, B=1/2/3, champion shapes K_rank ∈ {4,9}, planted dedup
collisions 0/5/12, mixed short-seq batches vk=[9,4,4] with fan_out 0
beyond vk_i, non-uniform mask + chosen_tok==0 false-match guard; per-seq
fan_idx formula ≡ concat of pre-M3 per-seq repeat_interleave incl.
zero rows at padded positions — design risk (a); per-seq glue blocks ≡
pre-M3 shared build per seq). M1 tests 9/9, M2 tests 8/8 OK
(individually; running m1+m2 in ONE unittest process fails 5 — a
pre-existing env-baking ordering artifact reproduced on the unmodified
tree, not an M3 regression). B=1 GPU regression smoke (champion config,
ns=4/out=128): 71.47 tok/s, L_p1 3.45, cache 0.81, zero Tracebacks —
matches M2's 71.35/3.49/0.83 within noise. Log:
experiments/proxy_async_overlap/b_gt1/m3_smoke/run.log.

## M4 — implemented (2026-07-18)

Landed per design §7 (`ssd/ssd/config.py` + `ssd/CLAUDE.md` invariant
line):

1. **Gate lift**: the DUET `max_num_seqs == 1` assert becomes `<= 8`
   (v1 cap — existing bs-bucket axis only, no new CG families). The
   stale comment blaming "Policy A accept_probs[0]" is replaced with the
   real history: the constraint was the single-seq Policy B pipeline
   (proxy wire / selector / phase-2 layout), batched in M1-M3.
2. **B==1-only gate guard** (§6): `duet_enabled` + `max_num_seqs > 1`
   now hard-errors (ValueError) at config time when any of
   `SSD_DUET_EXIT_TOPM_GATHER` / `SSD_DUET_EXIT_REPLICA` /
   `SSD_DUET_PROXY_ON_DRAFT` is set — fail fast instead of tripping the
   M1 runtime `assert B == 1` mid-run.

No engine-side fixes were needed: the two anticipated landmines did not
fire. bs=2 CG buckets captured cleanly for every family
(`fi_tree_decode`, `split_k1_long/short`, `split_k2`,
`duet_verify_k1/k2` — bucket axis {1,2} at max_num_seqs=2), and no
scheduler preemption appeared with the doubled lookahead reservations
(2048-token seqs, 517-block KV pool).

Validation: unit tests ssd/tests/test_b_gt1_m4.py — 6/6 OK (real
Config.__post_init__ on CPU, champion shapes: B ∈ {1,2,8} construct,
B=9 rejected, each gate raises at B=2 with all three still constructible
at B=1). M1 9/9, M2 8/8, M3 7/7 still OK individually.

GPU smokes (champion E9K24_jit, out=128, temp 0.7, GPUs 0-4):
B=2 `--b 2 --numseqs 8` (m4_smoke_b2/run.log), B=1 regression
`--b 1 --numseqs 4` (m4_smoke_b1/run.log). Both exit 0, zero
Tracebacks.

| metric | B=1 (ns=4) | B=2 (ns=8) |
|---|---|---|
| Decode TPS (aggregate) | 70.92 | 75.40 |
| Avg Cache Hits | 0.82 | 0.80 |
| Avg Phase 1 Accepted Len | 3.33 | 3.12 |
| Avg Tokens/step (incl recovery) | 3.61 | 2.75 |
| Phase 1 (draft) hit rate | 0.535 | 0.376 |
| Phase 2 (proxy) hit rate | 0.285 | 0.419 |
| Avg Phase 2 Accepted Len | 1.69 | 0.92 |
| Avg target full step (ms) | 55.95 | 80.54 |
| Avg draft step (ms) | 44.83 | 66.46 |

(P1 + P2 hit rates partition the cache-hit rate: 0.535+0.285=0.82 at
B=1, 0.376+0.419=0.80 at B=2.)

B=1 bar (TPS ≥ 68, L_p1 ≥ 3.0): PASS — 70.92/3.33/0.82 matches M3's
71.47/3.45/0.81 within the ns=4 noise band. B=2 bar (zero Tracebacks,
hits ≥ 0.70, L_p1 ≥ 3.0): PASS — 0.80/3.12, decode 75.40 aggregate.

Correctness read vs the design's "tok/step ≈ B=1" hope: hit rate does
NOT collapse (0.80 vs 0.82) and L_p1 holds (3.12 vs 3.33), but
tok/step drops 3.61 → 2.75 (−24%), concentrated in proxy-sourced rows:
the hit composition shifts from draft-sourced (0.535 → 0.376) toward
proxy-sourced (0.285 → 0.419) and P2 accepted len halves (1.69 → 0.92).
Aggregate decode TPS still rises +6.3% because two seqs verify per
step. Whether the P2 shift is a real B=2 effect or ns=8 prompt-mix
noise — and the per-seq perf story — is M5's sweep (B ∈ {1,2,4} vs
SD-best, interleaved).

## M5 — measured (2026-07-18)

Sweep: B ∈ {1,2,4}, DUET champion vs SD-best C (k7 f6), interleaved per
B, one run/cell, ns=20 out=256 seed 42, GPUs 0-4, ports 12900-12905.
Full tables + decomposition:
`experiments/proxy_async_overlap/b_gt1/m5_sweep/RESULTS.md`.

**The finding 5b hypothesis is REJECTED at v1.** Aggregate decode TPS:

| B | DUET | C | gap |
|---|---|---|---|
| 1 | 71.86 | 77.90 | −7.8% |
| 2 | 89.22 | 109.86 | −18.8% |
| 4 | 108.87 | 150.31 | −27.6% |

C scales ×1.93 B1→B4 vs DUET ×1.52. DUET's ingredients all landed —
hit 0.84 vs 0.74 at B=4, any-miss burden 0.50 vs 0.70 — and still
lost: (i) the M4 P2 flag is a REAL monotone B-effect (L_p2
1.64 → 0.85 → 0.49; P2 share of hits 35% → 53%), worth ~−10% tok/step
while C's tok/step is flat; (ii) DUET's step time grows faster than
C's on every axis (T_verify ×2.39 vs ×2.01, T_draft ×2.26 vs ×1.93)
despite 26 rows/seq vs 48 — vk_max padding, the mid-verify DUET block,
and 13 serial draft forwards crossing the tile cliff at B×16 rows;
(iii) any-miss JIT stalls are NOT the growing per-step term at B ≤ 4
(C absorbs a 0.70 any-miss burden with flat tok/step), so the
hit-rate advantage had nothing to amplify; JIT-short misses cap DUET's
miss rows at 1.48 tok (a token liability at B>1). Future-work levers
ranked in RESULTS.md §4.

**⚠ 2026-07-18 (M6): the M5 DUET numbers above were BUGGED.** The "P2
dilution" and miss-row collapse in this table were caused by a B>1
correctness bug (below), not by DUET's algorithm. Corrected sweep + the
revised verdict: §M6 below and `m5_sweep/RESULTS.md` (corrected table).

## M6 — B>1 short-row verify-window bug: root cause + fix (2026-07-18)

**Symptom** (M4/M5): L_p2 collapsed monotonically with B
(1.64 → 0.85 → 0.49) and miss-row tokens likewise (2.57 → 1.98 → 1.48),
while the P2 hit RATE went UP (0.28 → 0.445) — "keys match, chains
garbage". Physically impossible for per-seq-independent rollouts unless
something batched feeds wrong data.

**Root cause** (suspect #6 of the audit — NOT the M1/M3 batching math,
which a full-chain CPU test exonerated row-by-row): the target verify
input window. `prepare_decode_tensors_from_seqs` (runner_helpers.py)
builds each seq's verify rows as

    pos0 = seq.num_tokens - (k + 1)        # k = _duet_step_lookahead = vk_max

with the UNIFORM batch-level `vk_max`, but the speculator extended each
`seq.token_ids` by its PER-SEQ `vk_i`. For a short row (vk_i = K2 = 4)
in a mixed batch (vk_max = K1 = 9), pos0 lands `vk_max − vk_i = 5`
tokens too early: the 10-row window is `[5 stale context tokens | rec |
t1..t4]` instead of `[rec | t1..t9]`. Every `logits_p` row of that seq
is shifted by 5 positions, so the ratio test compares the P2/JIT chain
against the model's predictions of ALREADY-KNOWN context — near-certain
rejection at position 0 — and the recovery token is sampled from a
stale position (the model re-emitting an old context token, i.e. output
corruption, not just a perf loss). The guarding assert
(`num_cached_tokens == pos0`, runner_helpers.py L88) is stripped under
`python -O`, which every bench run uses.

**Why B=1 was immune**: a single-seq batch is always uniform —
vk_max = vk_i — so the window aligns. All M1-M3 B=1 regression smokes
passed for exactly this reason.

**Why the signature looked like "only seq 0 correct"**: it is actually
"short rows in MIXED batches are corrupted". Long rows (P1 hits,
vk_i = K1 = vk_max) are never shifted → L_p1 held/rose. Short rows
(every P2 hit AND every JIT-short miss row) are corrupted whenever the
batch also contains a long row — probability rising with B. The
dynamics are an attractor: a corrupted short row gets accept≈0 and a
degenerate recovery, its next-step request key (seq, 0, rec) lands in
the P2 pos-0 candidate fan → ANOTHER P2 hit (rate UP) whose chain is
again corrupted (L_p2 ≈ 0.1) — the seq churns short until a P1 hit
rescues it. This also explains M5's "curious" L_p1 rise (3.54 → 5.07
while P1 hit rate fell): degenerate repetitive text from corrupted rows
is easy to speculate deeply, and P1 hits self-select for the healthy
long-state seqs.

**Fix** (3 parts, all no-ops at B=1 / uniform batches):

1. `speculator_async.py` — `extend_seqs_for_verify` (extracted, unit
   tested): extend EVERY seq's token_ids by vk_max (short rows carry
   their padded tail) so pos0 = num_cached_tokens for every seq. Padded
   tails can never be accepted (M1 clamp) and the whole extension is
   rolled back by SpecDecodeStep's state restore.
   `num_draft_cached_tokens` keeps the per-seq vk_i + 1 advance.
2. `verifier.py _compute_and_send_proxy(valid_k=...)`: zero α̂ at a
   short seq's padded columns — h[b, vk_i] then carries the full
   all-real-accept mass and h beyond vk_i is 0, so chosen never lands
   at unreachable positions (the M1 design §3 claim, previously
   UNIMPLEMENTED — short seqs leaked P2 budget past vk_i).
3. `utils/verify.py`: residual (p−q)+ recovery adjustment now applies
   only below `min(K, valid_k)` — a clamped full-accept short row
   samples plain p at position vk_i (as the same event does at B=1)
   instead of subtracting a bogus uniform q from zero-padded logits.

**Validation**: unit tests `ssd/tests/test_b_gt1_m6_verify_window.py`
8/8 OK — (a) full draft-side chain (wire unpack → batched selector →
layout update → build args → merge keys) at B=3 with distinct per-seq
data: every row's (seq, k_idx, position, rope, seed) tuple ≡ the B=1
run of the same seq (suspects 1-5 clean); (b) the REAL
extend_seqs_for_verify + prepare_decode_tensors_from_seqs on a mixed
batch: window ≡ [rec]+spec[:vk_max] per seq, and the pre-M6 per-seq
extension provably slides the short row's window (prepare's own assert
fires); (c) verify(): padded columns that would ratio-accept are capped
by the clamp at vk_i and recovery comes from logits_p[b, vk_i]; (d)
proxy h-padding: a short seq's chosen_pos never exceeds vk_i (leaks to
6+ pre-fix). M1 9/9, M2 8/8, M3 7/7, M4 6/6 still OK.

GPU B=2 smoke (m4-smoke args, port 12910,
`experiments/proxy_async_overlap/b_gt1/m6_fix/b2_smoke/`): every
element of the signature reversed — L_p2 **1.75** (bugged 0.92; B=1
1.69), tok-on-miss **2.56** (bugged 1.98; B=1 2.57), P2 hit rate 0.286
(bugged inflation 0.419 gone), P1 hit 0.529 / L_p1 3.81, tok/step 3.80,
hits 0.82, decode TPS 101.5 (bugged 75.4 at the same cell), zero
Tracebacks.

**Corrected M5** (DUET cells re-run, same args/GPUs, ports 12911-13,
`experiments/proxy_async_overlap/b_gt1/m6_fix/duet_b{1,2,4}/`; C cells
unchanged — no DUET gates in them):

| B | DUET TPS | C TPS | gap (bugged) | L_p2 | miss-tok | P2 hit |
|---|---|---|---|---|---|---|
| 1 | 74.69 | 77.90 | −4.1% (−7.8%) | 1.73 | 2.59 | 0.269 |
| 2 | 104.59 | 109.86 | −4.8% (−18.8%) | 1.81 | 2.71 | 0.269 |
| 4 | 118.00 | 150.31 | −21.5% (−27.6%) | 1.63 | 2.68 | 0.274 |

L_p2 / miss-tok / P2-hit are now B-INVARIANT — the "monotone P2
dilution" of the original M5 was 100% bug artifact. B=2 is near-parity
with tok/step parity (3.89 vs 3.90). The surviving B=4 gap is ~85%
TIME-side (T_draft ×2.32 vs C ×1.93; T_verify ×2.46 vs ×2.01):
finding 5b stays unconfirmed, but for step-time-shape reasons
(draft forward count/width per B, vk_max-padded verify, mid-verify
block), not token/hit reasons. Full corrected tables + revised verdict:
`m5_sweep/RESULTS.md` correction section; docs/duet/12 B>1 section
rewritten accordingly.

## Verdict experiments — bug or physics? (2026-07-18)

Full writeup: `experiments/proxy_async_overlap/b_gt1/verdict/RESULTS.md`.
Two decisive experiments against the M6-corrected B=4 gap (−21.5%):

**Exp1 — B=4 PROFILE forensics** (champion args + SSD_PROFILE_DUET=1,
port 12920, 126.69 tok/s): every profile label checked against its
structural B×rows model. ALL match: phase1_replay 9 × 5.26 ms (64 rows
= 4 Marlin m-tiles; ×2.13 vs the 16-row 2.47 ms, BELOW the tile-linear
bound 5.79), phase2_replay 4 × 4.45 (40 rows), glue replay ×2.0,
preps/builds/merge flat (per-seq nested-mask build +0.2 ms — the M3
machinery is not a hidden cost), all-hit cache fill 0.89 ms ≡ B=1,
batched any-miss JIT 8.64 vs B=1's 8.00 ms (M2's latency-bound claim
confirmed at B×rows), walls label-accounted on both procs (no sync
storms). **Bug verdict: no remaining B>1 bug.**

Structural findings: (1) the TARGET binds — draft idle grew 6.0 → 34.5
ms/step (work 46.1 → 87.6 vs target wall 122.6), so the 13 serial
draft forwards never sit on the hit-step critical path (hit-step
spec_wait 3.0 ≈ B=1's 2.7); (2) width distribution: 93.3% of steps
dispatch K1-width verify (5.8% all-short K2, matching 0.447^4 theory)
while only 55% of rows are long → **vk_max padding = 17-21 ms/step**
(8.3 wasted rows × 2.23 ms/row marginal verify cost) — the dominant
time-side term; (3) the finding-5b miss-stall amplification term IS
present and grows with B (any-miss burden 0.57 vs C 0.70; 13-pt
frequency advantage, up from 6 pts at B=1; 7.8 ms/stall measured) but
is worth only +1..+5 ms/step. Decomposition sum (+12..+16 ms) closes
against the measured ΔT_target = +16.1 ms vs C.

**Exp2 — fat-shape retune probes** (B=4, PROFILE=0, ports 12921-2):

| cell | shape | serial fwds | verify rows | TPS | vs C | tok/step | t_step (ms) |
|---|---|---|---|---|---|---|---|
| champion | K1=9 K2=4 list [2×6,1×4] | 13 | 40 | 118.00 | −21.5% | 3.63 | 123.1 |
| fat7 | K1=7 K2=4 dfo=2 uniform | 11 | 32 | 144.72 | −3.7% | 3.71 | 102.5 |
| fat5 | K1=5 K2=4 dfo=3 uniform (--f 4) | 9 | 24 | **155.12** | **+3.2%** | 3.41 | 87.9 |
| C | k=7 f=6 | 7 | 32 | 150.31 | — | 3.99 | 106.2 |

fat7 lands T_verify at exact C parity (91.97 vs 91.42 — same 32-row
width) and its step is already faster than C's; fat5 — DUET's first
measured B>1 WIN — trades tokens (0.855 of C) for step time (0.827 of
C). The B=1 champion's deep-narrow shape was a tile-cliff artifact
that at B=4 paid K1-width verify padding on 93% of steps; the tile
cliff itself is amortized over seqs at B=4 (fat5's 72-row phase-1
forwards win anyway). Finding 5b: PARTIALLY CONFIRMED at v1 — B>1 is
DUET's winning regime once the shape is retuned per B. Caveats:
single run/cell (±4% token noise; +3.2% not band-clear alone — the
robust result is fat-beats-deep by +10..+31%), fat5 needs --f 4
(wider miss JIT), fat5 unmeasured at B ∈ {1,2}; B=1 champion remains
E9K24_jit.

## Per-B shape sweep + confirmed wins (pb_sweep, 2026-07-18/19)

Question: were fat5/fat7 (first guesses) actually optimal per B?
Full writeup: `experiments/proxy_async_overlap/b_gt1/pb_sweep/RESULTS.md`.
Grid per B (constraint K2≤K1, uniform dfo, f=dfo+pfo, ns=12 one
run/cell, C anchors rerun): B=4 9 shapes, B=2 5 shapes. All cells
rc=0 — the whole grid is inside the v1 constraint set.

**Answer: no.** At B=4 the surface keeps rising as K1 drops to the
grid edge: k3x3_d4p1 (K1=K2=3, dfo=4, k=6 f=5) 165.50 vs fat5's
149.72 (+10.6%). The response surface (scan):

- **K1 = verify width is the dominant knob, a pure time effect** —
  T_verify is a near-pure function of K1 (60.1/71.5/79/87 ms for
  K1=3/4/5/6 = B×2.25 ms/row, the verdict's marginal-row physics),
  while each K1 step buys only ~0.3 tok/step.
- **K2=K1 is free tokens** (zero vk_max gap → zero padding): k4x4 >
  k4x3, k7x6 > k7x4 at fixed K1, T_verify unchanged.
- **pfo=2** +2.7% mid-grid (draft idle funds it), neutral at the
  winner (k3x3_d4p2 166.27 ≈ 165.50).
- **dfo** flat at B=4; the main B=2 knob (dfo 2→3: +3.6%, hit
  0.81→0.84). B=2's surface is a flat ridge (±1.8%, every DUET cell
  beats C_b2): winner k6x5_d3p1 114.35 over k5x4_d3p1 114.22 is a
  coin-flip; k6x5_d3p1 is the one that got confirmed.

**Confirm phase (ns=20 out=256, 3-rep interleaved DUET/C per B) —
both winners band-clear:**

| B | shape | DUET mean (spread) | C mean (spread) | verdict |
|---|---|---|---|---|
| 4 | k3x3_d4p1 (k=6 f=5) | **169.42** (167.24-171.89) | 147.53 (142.48-151.28) | **+14.8%, band-clear** |
| 2 | k6x5_d3p1 (k=11 f=4) | **114.09** (112.82-115.77) | 106.73 (105.45-108.36) | **+6.9%, band-clear** |

B=4 mechanism: tok/step 2.85 vs 3.94 (0.723) × step time 67.4 vs
106.9 ms (1.586) — verify 16 rows vs C's 32, hit 0.87 vs 0.73. The
per-B optimum trend across the campaign: K1 9 → 6 → 3 and f 3 → 4 → 5
as B goes 1 → 2 → 4. **Finding 5b: CONFIRMED — the DUET win amplifies
with B (+0.5% → +6.9% → +14.8%) once the shape is retuned per B.**

Caveats: scan is single-run ns=12 (mid-grid orderings unresolved);
the C anchors + confirm ran a day after the DUET scan cells (a
run-script argparse bug crashed the original C cells pre-model-load;
confirm is internally interleaved, so the verdicts are drift-safe);
K1=3 is the grid edge (K1=2, K2>K1, B=8 unmeasured); the B=4 win is
100% step-time — regimes with costlier tokens shift the optimum back
toward depth. Recommended per-B configs: docs/duet/12 "B>1
recommended configs". **[07-19: the grid-edge caveat was closed by the
bscale campaign — next section.]**

## B-scaling campaign complete: B=8 + edges + figures (bscale, 2026-07-19)

The final gap-fill (`experiments/proxy_async_overlap/b_gt1/bscale/REPORT.md`):
a B=8 grid around the extrapolated optimum (K1 ∈ {2,3,4}, K1=K2), the
B=4 K1=2 edge cells the pb_sweep grid was missing, B=1 same-regime
anchors, and a 3-rep interleaved B=8 confirm. 11 scan cells + 6
confirm cells, all rc=0, zero Tracebacks — B=8 (the M4 gate cap) runs
the full DUET pipeline cleanly, CG bucket axis {1,2,4,8}.

**B=8 verdict: k2x2_d5p1 (K1=K2=2, dfo=5 pfo=1, k=4 f=6) beats C
band-clear — 210.39 vs 165.85 (+26.9%)**, spreads 209.74-211.11 vs
162.64-169.61 (worst DUET rep > best C rep by +23.7%). Mechanism:
tok/step 2.38 vs 3.83 (0.621) × t_step 90.4 vs 184.7 ms (2.044) →
R = 1.269. DUET verifies B×(K1+1) = 24 rows vs C's 64 (T_verify 80.7
vs 160.1 ms); hit 0.89 vs 0.73 → any-miss burden 0.62 vs 0.92 — at
B=8, C runs a JIT-degraded step 92% of the time (the M2 mixed
hit/miss fix is what lets DUET escape this). C saturates on the width
axis (B=4→8 only +12.4%, step time 107 → 185 ms) while DUET keeps
scaling (+24.2%).

**The complete amplification curve (finding 5b, final):**

| B | winner shape | vs C | status |
|---|---|---|---|
| 1 | E9K24_jit (K1=9 K2=4) | +0.5% (headline) / +0.6% (same-regime anchor) | parity |
| 2 | k6x5_d3p1 | +6.9% | band-clear |
| 4 | k3x3_d4p1 | +14.8% | band-clear |
| 8 | k2x2_d5p1 | +26.9% | band-clear |

**The shape law**: K1 9 → 6 → 3 → 2 (one grid step per B doubling),
K2 → K1 (uniform width, zero vk_max padding), f 3 → 4 → 5 → 6,
verify rows/seq 10 → 7 → 4 → 3. The bscale B=4 edge cells prove it is
a moving INTERIOR optimum, not "smaller is always better": K1=2 loses
at B=4 (157.3 vs 165.5 — the 7 ms step-time saving does not cover the
0.41 tok/step loss) and wins at B=8 (where the same step saves 19 ms).
Depth's token value is B-invariant; width's time cost is linear in B.

Deliverables: `bscale/REPORT.md` (full tables, mechanism, caveats) and
five figures (`bscale/figs/fig1..5`): TPS-vs-B, the amplification
curve, the shape law, the B=4 response surface with both edges, and
the per-seq throughput/latency tradeoff. Caveats worth remembering:
ns not a multiple of 8 at B=8 (tail steps below full width, identical
both sides); K1=1 and B>8 unmeasured (v1 cap); the B≥4 wins are 100%
step-time, so costlier-token regimes shift the optimum back toward
depth.
