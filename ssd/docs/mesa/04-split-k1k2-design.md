# Split-Only K1/K2 Mode — Design

## Motivation
Existing hybrid and legacy split paths both extend Phase 1 leaves via a
"continuation" pass — draft-sourced rows end up `K_long = K1 + K2` deep.
This document defines a new mode where draft-sourced and proxy-sourced rows
remain **independent**:

- Draft pass: K1 forwards → draft-sourced rows of depth **K1**
- Proxy pass: K2 forwards → proxy-sourced rows of depth **K2**
- Total draft work: **K1 + K2** forwards
- No continuation pass

The mode is gated by `SSD_FORCE_SPLIT_K1K2=1` and is implemented as a
separate path. Hybrid and legacy split remain untouched.

## Contract

### Bucket / valid_k space

| Mode             | valid_k space             | draft rows       | proxy rows |
|------------------|---------------------------|------------------|------------|
| hybrid           | `{K_long, K_short}` = `{K1+K2, K2}` | K_long          | K_short    |
| legacy split     | `{K_long, K_short}` (same) | K_long          | K_short    |
| **split K1/K2**  | **`{K1, K2}`**            | **K1**           | **K2**     |

**Supported scope: `K2 ≤ K1`.** Enforced by hard check in
`DraftRunner._init_prealloc_buffers` (raises `ValueError` if violated).
`K2 > K1` is unsupported because the proxy_horizon (= K2 positions in
draft phase 2) would exceed the accept_horizon (= valid_k ≤ K1 in target
verify), leaving `K2 - valid_k` proxy seed positions without a real
target source. Defining a tail source for that range is future work.

Within `K2 ≤ K1`:
- `K_max = K1`, `K_min = K2`.
- valid_k space = `{K1, K2}` (matched cache row depth).
- Phase 1 uses **long/short bucket** dispatch (mirrors hybrid):
  - `split_k1_long` (K=K1, pos=K1+1) for valid_k=K1 hit / first-step / miss
  - `split_k1_short` (K=K1, pos=K2+1) for valid_k=K2 hit
- Phase 2 uses single bucket `split_k2` (K=K2, pos=K2+1).

### Cache rows
- Draft-sourced row: written by Phase 1; payload depth = K1; valid_k = K1.
- Proxy-sourced row: written by Phase 2 proxy pass; payload depth = K2;
  valid_k = K2.
- Continuation rows: **forbidden in this mode.** No row is ever extended
  beyond its source phase's depth.

### Wire layout (next-step hit response)
- `cache_hits[B]`, `phase_source[B]`, `valid_k[B]`
- `out_tokens[B, K_max]` where `K_max = max(K1, K2)`
- `fan_out_tensor[K_max + 1]`
- `topk_ids[B, K_max, top_k]`, `topk_probs[B, K_max, top_k]`
- Pad rows whose valid_k < K_max with zero/sentinel beyond the row's
  actual depth.

### Verify CG family
Replace hybrid's `{verify_long, verify_short}` capture with
`{verify_k1, verify_k2}` for this mode:

- `verify_k1`: position_count = K1+1, MQ_LEN = `(K1+1) × max(dfo, pfo)`
- `verify_k2`: position_count = K2+1, MQ_LEN = `(K2+1) × max(dfo, pfo)`

Speculator dispatches by row's valid_k:
- valid_k == K1 → verify_k1 CG
- valid_k == K2 → verify_k2 CG

(Hybrid mode keeps its existing `verify_long/short` family — no shared
capture between the two modes.)

### Glue bucket (draft side, next-step hit)
Glue replays in `position_count = valid_k + 1`. Two buckets:
- `glue_k1`: position_count = K1+1
- `glue_k2`: position_count = K2+1

Dispatched by the matched cache row's valid_k.

### Phase 1 (Draft pass)
- Layout: forward_depth = K1, position_count = K1+1
  (NOT K_long+1 — there is no extension beyond K1.)
- Each forward extends draft tree by one depth.
- After K1 forwards, draft-sourced rows are written to cache with valid_k=K1.

### Phase 2 (Proxy pass)
- Independent of Phase 1's leaves.
- Layout: forward_depth = K2, position_count = K2+1.
- Seed: proxy-sourced tokens (sampled per proxy distribution received from
  target's early-exit logits).
- After K2 forwards, proxy-sourced rows are written to cache with valid_k=K2.

### merge_cache / next-step hit
- Reader picks a row whose first token matches the verified accept-path's
  next token (existing logic).
- The matched row may be draft-sourced (valid_k=K1) or proxy-sourced
  (valid_k=K2). Reader must propagate valid_k correctly downstream so
  the verify dispatcher picks the right CG.

## What to remove / forbid in this mode
- continuation pass (cont_replay, cont_prep)
- `split_cont_long / split_cont_short` layouts (any naming)
- `K_long` / `K_short` aliasing (in this mode's code paths only — hybrid
  may keep them)
- "plan-correct continuation mask" — never executed in this mode
- `split_via_hybrid` — semantics unrelated to this mode

## Reuse from hybrid
- valid_k tensor plumbing (per-row scalar passed from speculator → verifier)
- bucket dispatch infrastructure (CG family selection by valid_k)
- short/long verify capture pattern (capture two CGs, dispatch at run-time)

## Implementation order (per directive)
- **A.** core runtime rollback (DONE)
- **B.** contract definition (THIS DOC)
- **C.** split-only draft pass K1
- **D.** split-only proxy pass K2
- **E.** split-only glue bucket
- **F.** split-only verify bucket
- **G.** merge_cache / next-step hit path
- **H.** smoke validation (text correctness, no crashes, contract holds)
- **I.** benchmark (legacy split / true split K1/K2 / hybrid — three-way)

## Validation gates
Before "complete":
1. Draft rows valid_k == K1 in this mode (assert in cache writer / smoke)
2. Proxy rows valid_k == K2
3. No continuation rows produced (verify by event labels — cont_* events
   never fire when `SSD_FORCE_SPLIT_K1K2=1`)
4. Next-step hit handles both K1 and K2 rows without crash
5. Text output stable
6. Smoke test on 8B + benchmark on 70B AWQ
