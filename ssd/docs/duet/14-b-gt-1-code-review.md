# 14 — B>1 implementation code review (M1-M6)

**Date**: 2026-07-19. Scope: the engine changes of commits baa011c (M1),
af93cde (M2), 73fe75a (M3), 2cd2176 (M4), e5b586a (M6) reviewed against
docs/duet/13 (design + staged plan + verdict). Files:
`ssd/engine/{verifier,draft_runner,speculator_async,scheduler}.py`,
`ssd/engine/helpers/{cudagraph_helpers,tree_layout,runner_helpers}.py`,
`utils/verify.py`, `config.py`. Line numbers are HEAD@review-time
(b389476 parent c17b200).

**Headline**: no new high-severity correctness bug found. One
robustness fix applied (R1 — the exact `-O`-stripped assert class that
hid M6). Everything else is LOW/INFO with concrete deferred patches.

Verification: `tests/test_b_gt1_m{1,2,3,4}.py` +
`test_b_gt1_m6_verify_window.py` run individually — 38/38 OK before AND
after the applied fix; the hardened guard additionally verified under
`python -O` (fires on a pre-M6-shaped misaligned batch, passes aligned).

## Findings table

| # | Axis | Sev | Location | Finding | Action |
|---|---|---|---|---|---|
| R1 | robust | **MED** | runner_helpers.py:88 | `num_cached_tokens == pos0` assert stripped under `python -O` — guards silent output corruption (the M6 bug hid exactly here) | **APPLIED** (b389476): explicit `raise AssertionError`, ~B int compares/step |
| C1 | correct | OK | all reviewed files | No remaining B=1 / seq-0 indexing reachable at B>1 (full sweep below) | none |
| C2 | correct | LOW | verifier.py:454-456 + draft_runner.py:1564-1610 | Zero-score topk tie-break can put `chosen_pos > vk_i` entries on a short seq's wire; if that seq also has < total_budget positive-score entries, the selector takes them → phase-2 budget leaks to unreachable positions. Perf-only (keys `k_idx > vk_i` are never requested), B>1-mixed only | document; revisit only if L_p2 regresses at B>1 |
| C3 | correct | INFO | verifier.py:388-392, 444-447 | Short seq's all-real-accept event (h mass at col `vk_i < K`) draws candidates from the residual at the PADDED col `vk_i` (p_D = uniform from zero logits; token id 0 excluded) instead of the `pE_K`-style full-distribution treatment position K gets. Numerically ≈ top-k of p_E (uniform ≈ 1/V); v1-acceptable | document; v2 could gather pE at per-seq vk_i |
| C4 | correct | OK | — | Edge cases traced clean: all-short batch, all-miss batch, empty cache, B=8, B changing step-to-step, preemption re-admission, B=0 (details below) | none |
| C5 | correct | OK | speculator_async.py:13-47, utils/verify.py:140-172 | M6 × non-DUET/EAGLE: non-DUET wire carries uniform `valid_k = K` → extend-by-vk_max ≡ extend-by-K bit-identical; sync speculator passes `valid_k=None` → clamp + `_k_real` residual test reduce to pre-M6 forms; EAGLE extend data untouched | none |
| E1 | perf | OK | M1-M6 touched regions | Sync audit: no undocumented new GPU→CPU syncs. `extend_seqs_for_verify` REDUCED syncs 2B → 2 at B>1; the two `.max().item()` are documented swaps (verifier:114 replaces `torch.unique`, draft:546 replaces `valid_k[0].item()`) | none |
| E2 | perf | LOW | verifier.py:113-116 | The verify-path `valid_k.max().item()` sync is removable for free: speculator guarantees `speculations.size(1) == vk_max+1`, so `_step_lookahead = speculate_result.speculations.size(1) - 1` is sync-free and identical | defer (patch below; needs a GPU A/B before touching the hot path) |
| E3 | perf | LOW | draft_runner.py:1356 | `_irecv_duet_proxy` allocates a fresh `2·B·wire_N` int64 buf per step on the draft critical path (posted before glue) | defer: persistent buffer resized on B change |
| E4 | perf | LOW | cudagraph_helpers.py:324-328 | Non-bucket B (3,5,6,7): `active_cache_hits_list` len < padded `wrapper_bs` → falls back to `cache_hits[:B].tolist()`, one extra sync per phase per step-0. B ∈ {1,2,4,8} unaffected | defer: pad the threaded list to wrapper_bs instead |
| E5 | perf | OK | cudagraph_helpers.py:373-410 | Step-0 mask build is O(B·MQ·ctx) — linear in B, no O(B²); per-seq nested glue build is an O(B) list of `np.repeat`s (+0.2 ms measured at B=4, matches). Note `cache.clear()` on the MQ_LEN flip between phases means glue masks rebuild every phase-2 step-0 regardless of `_cached_fol` (pre-existing) | none |
| E6 | perf | INFO | draft_runner.py:1406,1433 | Fork selectors `clone()` `[B,P,V]` logits per step (~2.5·B MB) — pre-existing cost that now scales with B | note only |
| E7 | perf | INFO | draft_runner.py:421,492 | `match.float().argmax(dim=1)` computed twice per DUET step ([B,T] each, no sync) — `_hit_idx` could be reused in the fill block | defer, trivial |
| D1 | dead | INFO | verifier.py:421 | `proxy_fan_out_total` dead since Policy-B unification (pre-existing, predates M1 — left per surgical-change rule) | mention only |
| D2 | dead | INFO | verifier.py:407-415 | `cache_hits is not None and not config.jit_speculate` miss branch unreachable: `_compute_and_send_proxy` only runs under DUET and DUET config-requires `jit_speculate=True` (pre-existing) | mention only |
| D3 | dead | INFO | draft_runner.py:258-260 | Stale comment "B=1이므로 N = max_mq" contradicts the (already B-aware) `max_N = max_num_seqs * max_mq` line below it (pre-existing, rev1-era) | mention only |
| D4 | dead | INFO | draft_runner.py:1645-1650 | `metadata_ints` F (`_f0`, incl. the M3 nested-list special-case) is unpacked by `_decode_tree` but never consumed in the layout path | keep (3 lines; removing F changes the payload shape shared with the non-DUET path) |
| D5 | dead | INFO | speculator_async.py:194 | Fallback `int(valid_k.max().item())` unreachable (`_vk_max` is non-None whenever the branch is taken) — harmless defensive dead code from M6 | mention only |
| D6 | dead | OK | — | No M5-era debug prints/branches found in engine code (M5 was measurement-only; TRACE/PROFILE sites are env-gated and pre-existing) | none |
| D7 | dead | OK | tree_layout.py:48 + 2 consumers | `fan_idx_per_seq` dual-mode reviewed for simplification — verdict: KEEP (rationale below) | none |
| R2 | robust | LOW | model_runner.py:1111, draft_runner.py:1723 | `_step_lookahead/_step_valid_k in (K1, K2)` scalar asserts stripped under `-O`; a violation dispatches the wrong CG bucket. Failure is usually loud (CG shape mismatch), unlike R1 | defer-recommend: cheap hard raises |
| R3 | robust | OK | draft_runner.py:1548-1550, 1577-1585, 1636-1638 | Selector `__debug__` guards (chosen_pos range, Fix-③ per-seq `take.sum == total_budget`, fan_idx length) are GPU syncs — correctly debug-only. Fix-③'s invariant is config-guaranteed (proof sketch below), so the silent per-seq misalignment it would catch cannot occur with a well-formed wire | keep debug-only |
| R4 | robust | LOW | draft_runner.py:930 | `_glue_decode` B-consistency assert (`-O`-stripped); a mismatch means wire corruption but downstream `.view` fails loudly | keep as-is |

## Applied fix

**R1 — commit b389476** `fix(duet): harden verify-window pos0 guard`.
`prepare_decode_tensors_from_seqs` (verify path) now raises
`AssertionError` explicitly when `seq.num_cached_tokens != pos0`, so the
guard survives `python -O` (every bench run). This is the precise class
that hid M6: a misaligned window is *silent* corruption (shifted
logits_p rows + stale-position recovery), invisible to smoke metrics
until a sweep. Same exception type keeps
`test_b_gt1_m6_verify_window.test_pre_m6_extension_slides_short_row_window`
(assertRaises) unchanged. Cost: one Python int compare per seq per
verify step — noise. Validation: 38/38 unit tests OK after; `-O`
runtime check fires on a pre-M6-shaped batch and passes on an aligned
one.

## Axis 1 detail — the B=1-assumption sweep

Every `[0]` / seq-0 / scalar-collapse site in the reviewed files was
classified:

- **Config-gated unreachable at B>1** (M4 hard `ValueError` at
  `max_num_seqs > 1`): `policy_b_from_candidates` single-seq call
  (verifier.py:310-316), raw-proxy pack `exit_logits[0]`
  (verifier.py:349-373), `_policy_b_from_raw_proxy` `out_logits[0]`
  (draft_runner.py:1401), exit-replica / topm `orig_bs == 1` asserts
  (cudagraph_helpers.py:1317, 1359). The runtime asserts are
  `-O`-stripped but the config gate is the real guard — correct layering.
- **B==1-only profile branch** draft_runner.py:551-562 — explicitly
  branched on `B == 1`, with a batched else.
- **`layout.fan_out_list[0]`** sites: draft_runner.py:1740/1746 operate
  on the per-seq-UNIFORM phase-1 config list (not per-seq data);
  :1648-1650 feeds the dead metadata F (D4).
- **Batch-shape plumbing**: wire lengths (2·B·wire_N send =
  request-meta-B irecv, same step, same seqs), `[B,wire_N]` unpack
  views, per-seq accept clamp, per-seq h-padding, per-seq fan_out /
  fan_idx / masks — all traced consistent end-to-end.

**Edge cases (C4) traced**:

- *All seqs short*: `vk_max = K2` → k2 glue/verify buckets +
  `split_k1_short` layout; champion's short prefix `[2]*5` is uniform →
  uniform selector branch; `K_rank = K2` keeps `chosen_pos ≤ K2`.
- *All-miss batch / empty cache*: JIT-all with `_miss_vk` (K2 under
  `SSD_DUET_JIT_SHORT`), no overwrite pass; the one stale-KV corner
  (all-short dispatch + `_jit_K = K1 > vk_max`) is unreachable: a miss
  row without JIT_SHORT forces `vk_max = K1`, and an all-short all-hit
  batch runs no JIT.
- *B changes step-to-step (preemption/finish)*: every `[B,·]` stash is
  rebuilt or reallocated per step — speculator handshake bufs +
  speculations buf realloc on any B change (speculator_async.py:163,
  230); draft tree cache is reset+rebuilt every spec request
  (draft_loop:1975), so stale rows for departed seqs live ≤ 1 step and
  are keyed by monotonic `seq_id`; `split_k2_layout`'s mutated `[B,P]`
  fan tensors are overwritten before every consumer; re-admitted
  preempted seqs request key `k_idx = -2` (`last_spec_step_accepted_len
  = -1`) which can never match `fan_idx ≥ 0` → guaranteed clean miss.
  CG bucket padding (wrapper_bs > B) pads cache_hits with 0 and reuses
  the last real seq's glue block / block table; padded outputs discarded
  via slot_map -1.
- *B=8*: bucket axis covers it (M4 cap ≤ 8); handshake/spec buffers are
  sized by `max_num_seqs`.
- *B=0*: filtered at `SpecDecodeStep.decode`; `extend_seqs_for_verify`
  and the slicing block tolerate it defensively.

**M6 × EAGLE/non-DUET (C5)**: the async response wire always carries
`valid_k` (uniform `K` for non-DUET incl. EAGLE), so `extend by vk_max`
≡ `extend by K` and `num_draft_cached_tokens += K+1` — bit-identical to
pre-M6; the per-seq syncs it replaced are gone (E1). The sync
speculator's `SpeculateResult.valid_k` defaults to None → `verify()`
clamp skipped and `_k_real = K` reproduces the old `accept_until < K`
test exactly. EAGLE extend-count/acts plumbing is untouched by M6.

## Axis 2 detail — sync/allocation audit

Per-step GPU→CPU syncs in the touched regions, HEAD vs pre-M1:

| site | pre-M1 | HEAD | verdict |
|---|---|---|---|
| verifier lookahead | `torch.unique(valid_k)` + item | `valid_k.max().item()` | swap (E2: removable entirely) |
| draft dispatch scalar | `valid_k[0].item()` | `valid_k.max().item()` | swap |
| speculator extend | B×`.item()` + B×`.tolist()` | 1×`.tolist()` + 1×`.tolist()` | **improved** at B>1 |
| hit/miss branch | `.any()`/`.all()` | same | unchanged |
| phase-2 layout | 1×`fan_out.tolist()` | same, now [B,P] | unchanged (1 sync) |
| glue mask step-0 | `context_lens.tolist()` (+`cache_hits.tolist()` fallback) | same | unchanged; E4 fallback fires only at non-bucket B |
| selector | none (debug asserts gated) | same | unchanged |

New per-step allocations are all small and B-proportional: irecv buf
(E3, ≤1.5 KB at B=8), `_pad_cols` arange [K], h `[B,K+1]`, per-seq
fan_idx build (`arange.repeat(B).repeat_interleave`), nested
`fan_out_list` tolist (B×≤10 ints), per-seq numpy glue blocks. No
hidden O(B²) anywhere (E5): the only per-b loops are the pre-existing
mask/kv-meta builds, each O(B) with per-seq bodies independent of B.

### E2 deferred patch (verifier.py:113-116)

```python
# speculations is sliced to [B, vk_max+1] by SpeculatorAsync.speculate
# (M1 slicing block) — width IS vk_max+1, no sync needed:
if speculate_result.valid_k is not None and config.duet_phase1_k is not None:
    _step_lookahead = speculate_result.speculations.size(1) - 1
```

Identical value by construction; removes the last verify-path sync
before CG dispatch on the B=4-binding target. Not applied: hot-path
perf changes deserve their own GPU A/B, and the current sync is within
the documented budget.

## Axis 3 detail — `fan_idx_per_seq` dual-mode (D7)

Verdict: **keep**. The flag has exactly one setter
(`_update_phase2_layout_inplace`, the only runtime layout mutator) and
two 2-line consumer branches (`_build_tree_decode_args_for_layout`,
`_merge_and_populate_cache`). Collapsing the modes would mean either
(a) making the static phase-1 layouts per-seq too — pure churn, they
are per-seq-uniform by design — or (b) inferring per-seq-ness from
`fan_idx_hit.shape[0] == B*MQ_LEN`, which is implicit and fragile when
B·MQ coincidences occur. The explicit flag is the simplest correct
encoding; the split_k2 hit==miss aliasing (`fan_idx_miss =
fan_idx_hit`, `fan_out_list_miss = fan_out_list`) also lets the nested
glue build share one block list (cudagraph_helpers.py:350-352).

## Axis 4 detail — assert policy

Recommended policy for `-O`-stripped asserts in this code:

- **Hard-check (raise)**: guards whose violation is *silent corruption*
  and whose cost is CPU-scalar. Applied: R1 (pos0 window). Candidates
  if ever touched again: R2 bucket-dispatch scalars (violation is
  usually loud, so not applied now).
- **Keep `__debug__`-only**: anything that syncs the GPU
  (selector Fix-③ per-seq sums, chosen_pos range, fan_idx length,
  `_construct_tree_decode_args` N check). Fix-③'s invariant is
  mathematically covered by config sizing: wire entries at one position
  have distinct tokens, so dedup loss ≤ Σp fan_out_p = p1_sum, and
  `wire_N = total_budget + p1_sum + 2` (+ the `ceil(wire_N/(K_min+1))`
  top_k floor for short seqs) leaves ≥ total_budget + 2 valid entries
  per seq — the B>1 misalignment it would catch requires a malformed
  wire, not a reachable engine state.

## Deferred recommendations (ranked)

1. **E2** — drop the verify-path `.max().item()` via
   `speculations.size(1) - 1` (patch above) + GPU A/B at B=4.
2. **R2** — harden the two `in (K1, K2)` bucket-dispatch checks
   (scalar compares; symmetrical with R1's policy).
3. **E4** — pad `active_cache_hits_list` to `wrapper_bs` at the
   threading site to kill the non-bucket-B step-0 sync (only matters if
   B ∈ {3,5,6,7} cells are ever swept).
4. **E3** — persistent irecv buffer on the draft (resize on B change).
5. **E7** — reuse `_hit_idx` for the cache-overwrite gather.
6. **C3** — v2: per-seq `pE` gather at `vk_i` for the short-row
   all-accept candidate set (pairs naturally with any future per-seq
   vk_max-padding removal, the verdict's dominant B=4 cost).

Pre-existing dead code noted, deliberately untouched (surgical-change
rule): D1 (`proxy_fan_out_total`), D2 (non-jit miss branch under DUET),
D3 (stale B=1 buffer comment).
