# MESA Phase 2 Hybrid Implementation Plan

## Goal

Replace the current MESA Phase 1 (`K_long`-deep) + Phase 2 proxy
(`K_long`-deep) two-pass tree decode with:

- Phase 1: `K1`-deep tree decode producing draft-sourced seed
  sequences
- Phase 2: a single hybrid forward loop of depth `K2 = K_long - K1`
  that, in one batched forward per depth, processes both:
  - **continuation rows** — extend Phase 1 leaves by `K2` more
    tokens, reaching final draft-sourced suffix length `K_long`
  - **proxy-sourced rows** — independent `K2`-deep sequences using
    target-provided proxy tokens

The single hybrid loop is the only structural change — everything
else (cache, verifier, scheduler, etc.) is downstream wiring.

---

## Algorithm Semantics

This section is the single source of truth for what the algorithm
does. Every later section is derived from it. The terminology table
below is normative — the rest of the document uses these terms
consistently.

### Terminology

| Term | Meaning | Determined by |
|---|---|---|
| **fork position** | a slot in the speculative suffix where draft places `fan_out` alternative seed tokens | `valid_k` of the incoming hit |
| **seed token** | one of the per-position alternatives produced by Phase 1 or proxy | per-position `fan_out` |
| **forward depth** | how many model-forward steps a seed runs through to extend it into a sequence | config: `K1` for Phase 1, `K2` for Phase 2 (both continuation and proxy) |
| **MQ_LEN** | total batch size of seed tokens going through one tree decode pass | scales with `valid_k` (see below) |
| **row depth** | length of the speculative suffix stored in one cache row | config: `K_long` for draft-sourced, `K_short = K2` for proxy-sourced |
| **`valid_k`** | per-row stored row depth — `K_long` for draft-sourced, `K_short` for proxy-sourced and miss/JIT (v1) | per-row attribute |
| **`speculate_k`** | global max row depth | config = `K_long = K1 + K2` |

**Two axes of variation, kept separate:**

- **Forward depth (K1, K2) and row depth (K_long, K_short) are
  config-fixed.** They never vary per step.
- **MQ_LEN varies per step** because it scales with the incoming
  hit's `valid_k`. This is the only thing that changes across long-hit
  and short-hit cases on the draft side.

### Cache miss behavior — JIT produces short suffix in v1

When the cache lookup misses, `hit_cache_and_respond` falls through to
`jit_speculate(...)`, same as current MESA. The change in v1: JIT
runs the draft model sequentially for **`K_short` (= K2) steps** —
not `K_long` as current MESA does — producing a linear (no-fork)
suffix of length `K_short`.

**`valid_k` of the JIT-produced row = `K_short`.**

#### Why shorten miss JIT

JIT is sequential (autoregressive draft forwards), so it stalls the
async pipeline for the duration. With `K1 = K2 = K_long/2`, halving
the JIT depth halves the stall time and gets the engine back into the
overlapped Phase 1 / Phase 2 / target-verify pipeline faster. The
per-miss cost on a 70B AWQ stack drops from roughly `K_long ×
draft_forward_ms` to `K_short × draft_forward_ms`.

#### Trade-off

A miss step's maximum acceptance is now bounded by `K_short` tokens
(vs `K_long` previously). If miss-rate is small or Phase 2 proxy
already covers most miss cases at K_short anyway, the JIT-shortening
win dominates. If miss-rate is high *and* the draft model produces
useful suffixes well past `K_short`, this changes the wall-clock
math. v1 commits to the shorter JIT; revisit if measurements show
miss-step accept-rate dominating.

#### Wire / dispatch consequences

Miss rows feed into the same short-base path as proxy-sourced cache
hits:

- the JIT-produced row's verify uses `verify_short`
- the **next** step's glue / phase1 / phase2_hybrid use the short-
  base graph buckets (since the verifying suffix is `K_short` long)

No new graph bucket is needed — `verify_short`, `glue_short`,
`phase1_short`, `phase2_hybrid_short` are all already in the plan
for proxy-sourced hits and are reused for the miss path.

Phase 1 + Phase 2 hybrid still run on every step regardless of
hit/miss — they populate the **next** step's cache. Hit/miss only
determines what suffix is sent to *this* step's target verify.

### Per-step compute scales with `valid_k`

Glue / Phase 1 / Phase 2 hybrid all scale with the incoming hit's
`valid_k`. Specifically:

| Pass | MQ_LEN |
|---|---|
| Glue | `valid_k + 1` (linear forward) |
| Phase 1 | `(valid_k + 1) × mesa_draft_fan_out` |
| Phase 2 continuation | `(valid_k + 1) × mesa_draft_fan_out` (same seeds as Phase 1) |
| Phase 2 proxy | `proxy_fan_out_total = mesa_proxy_fan_out × (valid_k + 1)` |
| Phase 2 hybrid (continuation + proxy) | sum of the two above |
| Verify | 1 row × `valid_k + 1` (single suffix per step) |

**Note on proxy MQ_LEN — Policy A/B keeps dynamic per-position
allocation, all-accept slot included.** Current MESA's
`_compute_and_send_proxy` allocates `fan_out_list` over `valid_k + 1`
positions (`verifier.py:226`: `proxy_fan_out_total = mesa_proxy_fan_out
* (K + 1)`). The all-accept position (`p = valid_k`) does receive
budget; the tokens placed there come from draft logits (since proxy
residual is not sampled at all-accept) — see
`_select_proxy_sourced_tokens_policy_a` in `draft_runner.py:1080`.

v1 keeps this current behavior unchanged:

- per-position `fan_out_list[p]` is **dynamic** (Policy-dependent)
- total proxy MQ_LEN = `proxy_fan_out_total` is **fixed** by config
  and `valid_k`
- fan_out_list length: `valid_k + 1` (all-accept slot included, with
  draft-logits-sampled tokens)

Graph capture / wrapper sizing only cares about the total MQ_LEN, so
the per-position dynamism is invisible to it. The plan's MQ_LEN
formulas above use the total. Per-position layout details belong to
`HybridPhase2Plan`, which is built per step from the dynamically-
received `fan_out_list`.

**Future revisit**: it may be cleaner to drop the all-accept slot
from proxy entirely (since proxy = residual sampling, and the
all-accept position has no residual signal — the draft-logits tokens
placed there are a workaround that doesn't fit the proxy semantics).
v1 keeps current behavior; this is flagged as a follow-up cleanup.

So short-hit (`valid_k = K_short`) steps do strictly less work than
long-hit (`valid_k = K_long`) steps. This is the desired behavior —
the implementation is **not** padded to max compute.

### Draft-sourced row

A row produced by the draft side's own tree expansion. Forks at
`valid_k + 1` glue positions in Phase 1; each seed runs `K1` forwards
in Phase 1 and `K2` more forwards in Phase 2 continuation, reaching
final suffix length `K_long`.

- suffix length: `K_long = K1 + K2` (config-fixed)
- KV scratch: own Phase 1 KV slice + own A_tail slice
- prefix attended to during forward: persistent KV + glue KV + own
  Phase 1 KV + own A_tail prefix

### Proxy-sourced row

A row produced by an **independent proxy tree**. Rooted at the same
recovery token as draft-sourced rows but uses target-provided proxy
tokens. Each proxy seed runs `K2` forwards in Phase 2 proxy.

- suffix length: `K_short = K2` (config-fixed)
- KV scratch: own B_proxy slice only
- prefix attended to during forward: persistent KV + glue KV + own
  B_proxy prefix
- **does not read Phase 1 KV** — proxy is not a Phase 1 correction

### Proxy positions and the all-accept slot

Both Phase 1 and proxy cover `valid_k + 1` positions (depths
`0..valid_k`, including the all-accept slot at `valid_k`) — matches
current MESA. The distinction is *where the seed tokens come from*:

- Phase 1: tokens at every position picked by draft top-k from glue
  logits
- Proxy: tokens at positions `0..valid_k - 1` picked by `ĥ_i × top-k`
  of `(p_E - p_D)_+` residual; tokens at the all-accept position
  picked by draft top-k of glue logits at that position (current
  MESA's `_select_proxy_sourced_tokens_policy_a` already does this —
  v1 keeps this unchanged)

Sizes:

- Phase 1 fan-out list length: `valid_k + 1`
- Proxy fan-out list length: `valid_k + 1`
- Phase 1 MQ_LEN: `(valid_k + 1) × mesa_draft_fan_out`
- Proxy MQ_LEN: `(valid_k + 1) × mesa_proxy_fan_out`

Conceptually proxy-sourced rows from residual sampling only make
sense at reject positions (`0..valid_k - 1`), so it would be cleaner
to size proxy at `valid_k × pfo` instead. v1 stays with `(valid_k +
1) × pfo` to match current implementation; this is flagged as a
follow-up cleanup.

### Why the loop is "hybrid"

Continuation rows and proxy rows both need `K2` model forwards. They
could be run as two independent forward loops (continuation pass +
proxy pass), but that doubles Phase 2's compute by paying graph
launch / wrapper plan / mask precompute twice. The "hybrid" loop
packs both populations into one batch per depth and runs one forward
per depth.

This is the only algorithmic change. Phase 1 still produces the same
seed-token candidates as today; proxy still selects tokens by the
same residual-top-k rule; cache merge still produces the same kinds
of rows. The redesign is purely about how Phase 2 forwards happen.

### Incoming `valid_k` only varies the per-step batch shape

The `valid_k` of the incoming hit (= the cache row selected on this
step's verify) determines:

1. The verify graph dispatched (`verify_long` for `valid_k = K_long`,
   `verify_short` for `valid_k = K_short`)
2. The MQ_LEN of glue / Phase 1 / Phase 2 hybrid on the **next** draft
   step (because next-step glue runs over the suffix that was just
   verified, of length `valid_k + 1`)

Both effects are step-local. Forward depths and row depths are
unchanged. There is no "uniform / padded" mode in v1 — the engine
genuinely runs less work on short-hit steps.

---

## Scratch KV Layout

Five logical regions per draft step. The first is shared with the
persistent KV pool; the other four live in the speculative scratch.

| Region | Written by | Read by |
|---|---|---|
| persistent | (engine, accepted prefix) | all |
| glue | glue forward | Phase 1 forwards, Phase 2 continuation forwards, Phase 2 proxy forwards |
| Phase 1 KV | Phase 1 forwards | Phase 2 continuation forwards (own slice only) |
| A_tail | Phase 2 continuation forwards | self (own slice only, while extending) |
| B_proxy | Phase 2 proxy forwards | self (own slice only, while extending) |

Per-row block tables therefore differ by row kind:

- continuation row: persistent pages + glue pages + own Phase 1
  pages + own A_tail pages
- proxy-sourced row: persistent pages + glue pages + own B_proxy
  pages

There is no cross-row sharing within Phase 1 KV / A_tail / B_proxy.
Slot pools for the three speculative regions are disjoint by
construction.

### Buffer sizing

Phase 1 KV is retained during Phase 2 (continuation reads it), so
the new layout uses more pages per spec step than current MESA. v1
just accepts the increase. Implementation tasks:

- `compute_megaspec_lookahead()` — re-derive worst-case
  (`valid_k = K_long`) including all four speculative regions
- `_init_prealloc_buffers` — `_spec_logits_buf`, `_spec_tokens_buf`,
  `_spec_activations_buf` sized by long-hit hybrid combined MQ_LEN
- block-manager reservation per draft sequence — confirm no overflow

No design comparison to current MESA peak; the new layout simply
needs enough pages, which v1 budgets max-case.

---

## Wire Contracts

### Speculator response (draft → speculator_async)

Existing fields plus per-row `valid_k`:

```
fused_response: [B, cache_hits | phase_source | valid_k | tokens(K_long)]
```

Variable-shape sends are not used; `tokens` is always `K_long`-sized.
Draft fills only the meaningful prefix of length `valid_k`; tail
positions of `tokens` are undefined.

### Proxy payload (target → draft, mid-verify)

Fixed max-size payload sent every step:

```
[K_long + 1]        int64s for fan_out_list
[B, K_long, top_k]  int64s for topk_ids
[B, K_long, top_k]  int64s for topk_probs
```

Target side computes meaningful values for `valid_k` positions only
(`exit_logits[:, :valid_k, :]`, residual top-k over `valid_k`
positions, `fan_out_list[:valid_k + 1]`). The remaining `K_long -
valid_k` positions are sent as undefined / zero. Both sides know
`valid_k` independently before the NCCL op (target via the row it's
verifying, draft via cache lookup), so no size handshake is needed.

Draft consumes the prefix:

```
fan_out_list[:valid_k + 1]
topk_ids[:, :valid_k, :]
topk_probs[:, :valid_k, :]
```

The bandwidth waste on short-hit steps is ~1 KB at B=1 (negligible).
Max-size keeps the sender / receiver code simple — no per-step
slicing on the wire, no two-phase NCCL.

---

## Graph Strategy — 8 buckets

Every CUDAGraph family that captures a forward over MQ_LEN tokens has
one bucket per incoming `valid_k`:

| Family | Buckets | MQ_LEN (long / short) | Forward depth |
|---|---|---|---|
| glue | `glue_long`, `glue_short` | `K_long+1` / `K_short+1` | 1 (linear forward) |
| phase1 | `phase1_long`, `phase1_short` | `(K_long+1)×fo` / `(K_short+1)×fo` | `K1` |
| phase2_hybrid | `phase2_hybrid_long`, `phase2_hybrid_short` | `(vk+1)×(fo + pfo)` | `K2` |
| verify | `verify_long`, `verify_short` | 1 query × `K_long+1` / 1 × `K_short+1` | (target attention) |

Eight graphs total, captured at engine init.

**Per-step runtime cost of having 8 graphs vs fewer = 0.** Graph
selection is a Python dict lookup (`graphs[key].replay()`); replay
cost is independent of how many graphs exist. The only added cost is
capture time at startup (one-time, seconds-to-minutes scale).

---

## `HybridPhase2Plan` Dataclass

Per-step plan object built once after proxy arrives. Indexed in the
hot path by depth without any further Python work.

Suggested file: `ssd/ssd/engine/helpers/hybrid_phase2_plan.py`.

### Fields

#### Identification

- `valid_k: int` — the incoming row's `valid_k`. Picks the graph
  bucket.
- `name: str` — `"phase2_hybrid_long"` or `"phase2_hybrid_short"`
- `graph_key: str` — matches the captured CUDAGraph name
- `k2: int` — forward depth (= config `mesa_phase2_k`)

#### Row partition

For B=1 the batch has one block of each kind:

- `cont_row_count: int = (valid_k + 1) × mesa_draft_fan_out`
- `proxy_row_count: int = mesa_proxy_fan_out × (valid_k + 1)` (=
  `proxy_fan_out_total`; per-position breakdown dynamic per Policy
  A/B but total fixed; all-accept slot included to match current
  MESA — see Algorithm Semantics for the v1-vs-future-cleanup note)
- `total_row_count: int = cont_row_count + proxy_row_count`
- `phase_split_offset: int = cont_row_count` — index where proxy
  rows start in the flat hybrid batch

#### Row classification (small int8 tensors)

- `per_row_region_id: torch.Tensor [total_row_count]` — `0` =
  continuation, `1` = proxy-fresh
- `per_row_valid_source_kind: torch.Tensor [total_row_count]` —
  output cache row's `valid_k` (continuation rows → `K_long`, proxy
  rows → `K_short`)

#### Per-row × per-depth attention plumbing

These are the buffers that let the depth loop avoid any Python-side
tensor construction. All are filled once per step in `phase2_build`:

- `per_row_context_lens_by_depth: torch.Tensor [total_row_count, K2]`
  — absolute KV length each row's depth-`d` query attends to
- `per_row_slot_maps_by_depth: torch.Tensor [total_row_count, K2]` —
  paged-KV slot where the depth-`d` token's KV is written
  (continuation → A_tail, proxy → B_proxy; pools are disjoint)
- `per_row_kv_indptr_by_depth: torch.Tensor [total_row_count + 1, K2]`
- `per_row_kv_indices_by_depth: torch.Tensor [K2, max_total_indices]`
  — depth-major; row `i`'s pages at depth `d` =
  `[d, indptr[i,d]:indptr[i+1,d]]`
- `per_row_block_tables: torch.Tensor [total_row_count,
  max_pages_per_row]` — per-row block table (continuation rows
  cover persistent + glue + own Phase 1 + own A_tail; proxy rows
  cover persistent + glue + own B_proxy)

#### Custom mask precompute

The current FlashInfer tree-decode path's mask builder assumes a
uniform tree; hybrid Phase 2 has two row populations with different
prefix shapes. A new helper `build_hybrid_packed_mask(plan)` (in
`cudagraph_helpers.py`) emits:

- `per_depth_packed_masks: torch.Tensor [K2, max_packed_mask_size]`
- `per_depth_mask_indptr: torch.Tensor [K2, total_row_count + 1]`

Mask shapes per row:

- continuation row at depth `d`: persistent + glue + own Phase 1 KV
  + own A_tail prefix (`d` tokens)
- proxy row at depth `d`: persistent + glue + own B_proxy prefix
  (`d` tokens)

No inter-row sharing in the mask.

#### Position math (same philosophy as `TreeLayout`)

- `cont_initial_positions / cont_initial_rope_positions`
- `proxy_initial_positions / proxy_initial_rope_positions`
- `step_pos_offsets: [K2] = arange(K2)`
- `step_rope_offsets: [K2] = arange(K2)`

The depth loop computes `initial[:, None] + step_offsets[None, :]`
in one elementwise op rather than rebuilding tensors.

### Single instance with max-size buffers

There is **one `HybridPhase2Plan` instance per `DraftRunner`**, not
two. All tensor fields are allocated once at engine init at max size
(= sized for the long bucket: `cont_row_count = (K_long+1) ×
fan_out`, `proxy_row_count = (K_long+1) × pfo`).

`phase2_build` fills the buffers in-place per step. Scalar fields
(`valid_k`, `cont_row_count`, `proxy_row_count`, `total_row_count`,
`name`, `graph_key`) are recomputed per step from the incoming
`valid_k`.

The hybrid CUDAGraph (long vs short) is selected at runtime via
`graph_key`. The captured graph itself depends on the bucket; the
plan instance is shared.

No new GPU allocation in the hot path.

---

## Per-Phase Walkthrough

This section walks through one MESA step, naming exactly what runs.
This is the engine's behavior on a long-hit step; short-hit is
identical with smaller MQ_LEN values.

### Step start: speculate

Draft serves the spec request from cache, returning one row per seq
(B=1). The selected row's `valid_k` (= `K_long` if draft-sourced, =
`K_short` if proxy-sourced) is included in the response.

### Glue decode

```
input  : [B, valid_k + 1]  — recovery token + valid_k speculative tokens
output : logits at valid_k + 1 positions
graph  : glue_long or glue_short (dispatch by valid_k)
KV     : writes glue region (linear chain of valid_k + 1 slots)
```

### Phase 1 tree decode

Phase 1 takes glue logits at `valid_k + 1` positions and selects
`mesa_draft_fan_out` candidate tokens at each. These become the seed
tokens for `K1` forwards.

```
input  : [B, MQ_LEN_phase1]  — MQ_LEN_phase1 = (valid_k+1) × fan_out
output : K1-token sequences for each seed slot
graph  : phase1_long or phase1_short (dispatch by valid_k)
KV     : reads persistent + glue; writes Phase 1 KV (own slot per seed)
```

### Proxy unpack (parallel with Phase 1)

Draft `irecv`s the max-size proxy payload, slices the meaningful
`:valid_k`-prefix, and feeds it into proxy seed selection
(`_select_proxy_sourced_tokens_policy_a` — unchanged logic, just
iterates `valid_k` positions instead of `K`).

### Phase 2 hybrid build

`_build_phase2_hybrid_plan` fills the `HybridPhase2Plan` in-place:

- continuation rows: one per Phase 1 seed slot (= `(valid_k+1) ×
  fan_out` rows). Input token at Phase 2 depth 0 = the last token
  produced by Phase 1 forward (K1-th token of the seed's sequence).
  Output region = A_tail.
- proxy rows: one per proxy seed slot (= `(valid_k+1) × pfo` rows
  per the closed payload contract — all-accept slot included). Input
  token at Phase 2 depth 0 = proxy-selected token. Output region =
  B_proxy.
- per-row × per-depth tensors filled in-place (block tables, slot
  maps, context lens, kv indptr / indices, packed masks)

### Phase 2 hybrid replay

```
input  : [B, MQ_LEN_hybrid]  — MQ_LEN_hybrid = (valid_k+1)×(fo + pfo)
output : K2-token sequences for each row in the hybrid batch
graph  : phase2_hybrid_long or phase2_hybrid_short
KV     : continuation rows write A_tail; proxy rows write B_proxy
```

### Cache emit

Two row kinds emitted into `tree_cache`:

- `(valid_k + 1) × fan_out` draft-sourced rows of suffix length
  `K_long` (Phase 1 K1 tokens + Phase 2 continuation K2 tokens)
- `(valid_k + 1) × pfo` proxy-sourced rows of suffix length `K_short = K2`

Each row's `valid_k` field is set per `per_row_valid_source_kind`.

### Verify (next step)

Target picks the suffix of one selected row from the cache and runs
either `verify_long` (selected row was draft-sourced) or
`verify_short` (selected row was proxy-sourced). One verify per
step; never two.

---

## Implementation Sequence

Vertical slices: end-to-end MESA still runs after each phase.

### Phase 0 — `valid_k` plumbing (no algorithm change)

- add `mesa_phase1_k`, `mesa_phase2_k` to config (validation only;
  not yet consumed)
- extend `SpeculateResult` with `valid_k`
- extend NCCL fused response with `valid_k (B)` — adds B int64s to
  the existing fused payload
- engine still treats every row as `valid_k = K_long`; the new
  field is plumbed but its value is uniform

### Phase 1 — cache + valid_k plumbing

- `tree_cache_tokens / logits` shaped `K_long`
- `tree_cache_valid_k` per-row tensor
- glue / verify still uniform `K_long+1` (no dispatch yet)
- still no algorithmic change visible in TP

### Phase 2 — config + TreeLayout extension (no functional change)

- add `mesa_phase1_k`, `mesa_phase2_k` config validation (sum =
  `speculate_k`)
- extend `TreeLayout` with `position_count` separate from `K`
  (forward_depth) — non-MESA call sites still satisfy `position_count
  = K + 1` so they're unchanged
- engine still runs current MESA: Phase 1 deep `K_long`, Phase 2
  proxy deep `K_long`. Cache rows still all `valid_k = K_long`.
- this phase only lands **infrastructure**, not algorithmic split.
  The actual K1/K2 split and continuation wiring lands in Phase 3.

### Phase 3 — Phase 1 K1 split + Hybrid Phase 2 (long-base bucket only)

- introduce `phase1_layout_long` (forward depth `K1`, position
  count `K_long + 1`) and use it instead of the current Phase 1 layout
- Phase 1 now emits seed sequences of length **K1**, not `K_long`
- introduce `HybridPhase2Plan` (long-base bucket only)
- one-loop hybrid replay of depth `K2`: continuation rows extend
  Phase 1 K1-length sequences by K2 more tokens; proxy rows produce
  K2-deep independent sequences
- 5-region scratch partition lands here
- proxy rows attend to persistent + glue (not Phase 1 KV)
- cache emit: draft-sourced rows of suffix length `K_long`
  (= K1 + K2), proxy-sourced rows of suffix length `K_short = K2`
- engine still treats every row as `valid_k = K_long` for verify
  dispatch (heterogeneous valid_k lands in Phase 4)
- this is where the long-hit forward-work reduction first appears

### Phase 4 — verify dispatch (heterogeneous valid_k)

- proxy-sourced rows stored with `valid_k = K_short`
- verifier dispatches `verify_long` / `verify_short`
- metrics normalize by `valid_k`

### Phase 5 — short-base graph buckets

- generalize glue / phase1 / phase2_hybrid to runtime `valid_k`
- capture `glue_short`, `phase1_short`, `phase2_hybrid_short`
- runtime selects all 8 buckets
- short-hit forward-work reduction lands here too

### Phase 6 — validation

- confirm capture of all 8 graph families
- benchmark draft total forward work vs current MESA
- record per-phase breakdown (existing `mesa_*` profiling labels)

---

## File-by-File Changes

### `ssd/config.py`

- add `mesa_phase1_k` (= `K1`), `mesa_phase2_k` (= `K2 = K_short`)
- validate `K1 > 0`, `K2 > 0`, `K1 + K2 == speculate_k`
- update `mesa_proxy_top_k` auto-raise bound: `pfo * (K_long + 1) +
  dfo + 2` (worst case = long-hit; replaces existing `pfo*(K+1) + dfo
  + 2`)

### `ssd/engine/helpers/speculate_types.py`

- extend `SpeculateResult` with `valid_k: torch.Tensor [B]`

### `ssd/engine/speculator_async.py`

- enlarge fused response to include `valid_k`
- max-sized token / logit buffers (`K_long`)
- pass `valid_k` into `SpeculateResult`

### `ssd/utils/async_helpers/async_spec_helpers.py`

- `compute_megaspec_lookahead()`: re-derive per "Scratch KV Layout"
- `make_glue_decode_input_ids(...)`: add `valid_k` parameter; returns
  `[B, valid_k + 1]`
- `prepare_glue_decode_ctxt(...)`: take `valid_k`

### `ssd/engine/draft_runner.py`

Main implementation file.

- add `phase1_layout_long`, `phase1_layout_short` (depth `K1`,
  `MQ_LEN = (valid_k+1) × fan_out`)
- add `HybridPhase2Plan` plumbing (`_build_phase2_hybrid_plan`,
  `_decode_phase2_hybrid`, `_merge_and_populate_cache_hybrid`)
- replace existing `proxy_layout` usage in the MESA path
- preserve current non-MESA path unchanged
- `_glue_decode(...)`: dispatch `glue_long` / `glue_short` by
  `valid_k`
- `_irecv_mesa_proxy` / `_unpack_mesa_proxy`: unchanged max-size
  buffers; per-step slice `:valid_k` prefix
- `jit_speculate(...)` and `hit_cache_and_respond(...)` miss path:
  call JIT with depth `K_short` (= K2), not `K_long`. Return
  `valid_k = K_short` for the JIT row (see "Cache miss behavior")
- `_select_proxy_sourced_tokens_policy_a`: iterate `valid_k`
  positions
- `_init_prealloc_buffers`: extend max-MQ_LEN allocation to cover
  both long-base and short-base hybrid buckets; allocate
  `HybridPhase2Plan` buffers (per-row × per-depth tensors, packed
  masks)
- `hit_cache_and_respond(...)` returns `valid_k` per row (already
  partly done by the existing phase_source instrumentation; extend)
- `_service_spec_request(...)` packs `valid_k` into the fused
  response

### `ssd/engine/helpers/tree_layout.py`

**Contract change required.** The current `TreeLayout` packs three
tightly-coupled values into a single `K`:

- forward depth (used as `arange(K)` for `step_pos_offsets`,
  `step_rope_offsets`)
- position count (`len(fan_out_list) == K + 1` invariant)
- depth used in `metadata_ints` and the `+ (K + 1)` glue position
  offset in `_build_tree_decode_args_for_layout`

The new design needs **forward depth and position count separated**:
Phase 1 has `forward_depth = K1` (config-fixed) but `position_count
= valid_k + 1` (variable per step). The `len(fan_out_list) == K + 1`
invariant is broken.

v1 picks **option A — extend `TreeLayout`**:

- rename the existing `K` field semantically to `forward_depth`
  (keep the attribute name `K` for backward compat in non-MESA call
  sites, where `K + 1 == len(fan_out_list)` invariant continues to
  hold)
- add `position_count: int` field (= `len(fan_out_list)`)
- audit `_build_tree_decode_args_for_layout(...)`:
  - `metadata_ints` depth → `forward_depth`
  - `+ (K + 1)` glue offset → `+ position_count`

For non-MESA call sites the two are still `K` and `K + 1`
respectively; nothing visible changes. For MESA Phase 1 layouts they
are different (`K1` and `valid_k + 1`).

Two Phase 1 instances created at engine init:

- `phase1_layout_long`:  `forward_depth = K1`, `position_count = K_long + 1`
- `phase1_layout_short`: `forward_depth = K1`, `position_count = K_short + 1`

Hybrid Phase 2 does not use `TreeLayout` — it uses
`HybridPhase2Plan`.

### `ssd/engine/helpers/hybrid_phase2_plan.py` (new)

- `HybridPhase2Plan` dataclass per the fields listed earlier
- `build_hybrid_packed_mask(plan)` helper

### `ssd/engine/helpers/cudagraph_helpers.py`

- `capture_glue_cudagraph(model_runner, valid_k)` — call twice
  (long, short)
- `capture_phase1_seed_cudagraph(model_runner, valid_k)` — twice
- `capture_phase2_hybrid_cudagraph(model_runner, valid_k)` — twice
- adapt the existing verify capture to take `valid_k` — twice
- new hybrid mask builder consuming `HybridPhase2Plan`
- step-0 precompute philosophy preserved

### `ssd/engine/model_runner.py`

- add new FlashInfer wrapper families (one per graph bucket):
  `phase1_long`, `phase1_short`, `phase2_hybrid_long`,
  `phase2_hybrid_short`. `custom_mask_buf` sized for the hybrid
  mask, not for a uniform tree.
- the existing legacy `draft` / `proxy` wrappers from the old MESA
  path can be removed once hybrid lands
- capture and store all 8 graph families
- runtime dispatch by selected row's `valid_k`
- non-MESA paths untouched

### `ssd/engine/verifier.py`

- `_compute_and_send_proxy(...)`: compute correction over `valid_k`
  positions of the row being verified; send max-size payload
  (`K_long`-sized) per "Wire Contracts". Read `exit_logits[:,
  :valid_k, :]`, residual top-k over `valid_k` positions, fill
  `fan_out_list[:valid_k + 1]`; pad rest with zeros.
- `verify(...)`: consume `valid_k` from `SpeculateResult`; dispatch
  `verify_long` / `verify_short` accordingly
- reshape target outputs by `valid_k + 1`
- metrics normalize by `valid_k`. Existing `phase1_hits` /
  `phase2_hits` instrumentation unchanged.

### `ssd/engine/scheduler.py`

- update `draft_lookahead_len` to consume the new
  `compute_megaspec_lookahead()` value
- block manager reservation per draft sequence stays max-case; the
  new layout **uses more pages per spec step** than current MESA
  (Phase 1 KV must persist through Phase 2 — see "Buffer sizing"),
  so the lookahead value grows. v1 simply accepts this.

### `ssd/engine/llm_engine.py`

- existing `phase1_hits` / `phase2_hits` printing remains
- no new metrics required for v1; the existing per-phase profiling
  labels (`glue`, `phase1_replay`, `phase2_replay`, `merge_cache`,
  etc.) carry over

---

## Performance Estimate

Setup: `K_long = 8`, `K1 = K2 = 4`, `fan_out = pfo = 2`.

Forward work counted as `MQ_LEN × forward depth` (one query token
forwarded per slot per depth). Verify is 1 row × `valid_k + 1` for
B=1.

### Long-hit step (`valid_k = K_long = 8`)

| Pass | MQ_LEN | depth | row-depths |
|---|---:|---:|---:|
| Glue | 9 | 1 | 9 |
| Phase 1 | (8+1)×2 = 18 | 4 | 72 |
| Phase 2 continuation | 18 | 4 | 72 |
| Phase 2 proxy | (8+1)×2 = 18 | 4 | 72 |
| Verify long | 1 | 9 | 9 |
| **Sum** | | | **234** |

### Short-hit step (`valid_k = K_short = 4`)

| Pass | MQ_LEN | depth | row-depths |
|---|---:|---:|---:|
| Glue | 5 | 1 | 5 |
| Phase 1 | (4+1)×2 = 10 | 4 | 40 |
| Phase 2 continuation | 10 | 4 | 40 |
| Phase 2 proxy | (4+1)×2 = 10 | 4 | 40 |
| Verify short | 1 | 5 | 5 |
| **Sum** | | | **130** |

### Comparison vs current MESA

Current MESA at `K = K_long = 8`, same fan-outs:

| Pass | MQ_LEN | depth | row-depths |
|---|---:|---:|---:|
| Glue | 9 | 1 | 9 |
| Phase 1 | 18 | 8 | 144 |
| Phase 2 proxy | 18 | 8 | 144 |
| Verify | 1 | 9 | 9 |
| **Sum** | | | **306** |

Reductions:

- long-hit step: 306 → 234 ≈ **−24%**
- short-hit step: 306 → 130 ≈ **−57%**
- 50/50 mix average: ≈ **−40%**

### Wall-clock estimate (rough)

Current 70B AWQ MESA best-config: draft ≈ 49 ms, verify ≈ 47 ms,
TP ≈ 72 tok/s.

With ~10 ms fixed graph-replay overhead per step (glue + phase1 +
phase2_hybrid replay launches), the remaining ~39 ms is forward work
that scales with row-depths.

- new draft forward (avg) ≈ `39 × 175 / 297 ≈ 23 ms`
  (`175` = mix avg of `(234-9, 130-5)/2 = (225+125)/2`; `297` =
  current draft forward only = `306 - 9 verify`)
- new draft step ≈ `23 + 10 = 33 ms`
- new verify avg ≈ `47 × (9 + 5) / 18 ≈ 36 ms` (long/short
  averaged)
- new step ≈ `max(33, 36) + handshake` ≈ 40-45 ms
- expected TP ≈ **100-115 tok/s** (up from 72)

**Conservative target: +35-55% TP.**

### Where the estimate can be wrong

- per-depth Python overhead in hybrid Phase 2 if `HybridPhase2Plan`
  is not fully precomputed (Risk #2)
- if `K2` is too short for proxy-sourced rows to reach useful cache
  hits, accept rate drops and TP gain shrinks
- async pipeline overlap means the slower side dominates — if draft
  drops below verify, only verify-side savings count for total TP

---

## Testing Plan

### Unit / structural

- `mesa_phase1_k > 0`, `mesa_phase2_k > 0`, sum equals `speculate_k`
- cache row stores correct `valid_k` per source
- max-size proxy payload — sender pads, receiver consumes
  `valid_k`-prefix
- `HybridPhase2Plan.total_row_count` matches `(valid_k+1) × (fan_out
  + pfo)`

### Draft correctness

- Phase 1 + Phase 2 continuation reproduce, for each draft-sourced
  row, the same suffix as a uniform `K_long`-deep tree decode would
  have produced (numerical match within tolerance)
- Phase 2 proxy reproduces, for each proxy-sourced row, the same
  suffix as a standalone `K2`-deep proxy tree decode (current MESA's
  proxy path at `K = K2`)
- **hybrid vs split equivalence**: hybrid batch result = continuation-
  only forward + proxy-only forward run separately and concatenated
  (this is the catch-all correctness test)
- region isolation: continuation rows write only A_tail; proxy rows
  write only B_proxy; slot pools are disjoint

### Graph capture / replay

- all 8 graph families capture without error
- per-step dispatch picks the right bucket by `valid_k`
- replay produces identical outputs to eager forward at the same
  inputs

### End-to-end MESA

- runs Policy A and Policy B without deadlock or graph mismatch
- generated text quality is stable vs current MESA on a fixed
  prompt set

### Performance

- record the existing `mesa_*` profiling labels; confirm Phase 2 is
  one replay per depth, not two
- benchmark TP vs current MESA on the 70B AWQ + TinyLlama AWQ stack
- verify the long-hit / short-hit forward-work reduction is visible
  in the per-phase breakdown

---

## Risks

### 1. Per-row block-table / mask correctness

The hybrid Phase 2's per-row attention plumbing (block tables, kv
indices, packed masks) is the new runtime-correctness surface. A
small error here can produce silent wrong outputs (attention reading
the wrong KV pages).

Mitigation: the hybrid-vs-split equivalence test in "Testing Plan"
catches this directly.

### 2. Per-depth Python overhead

If `HybridPhase2Plan` is not fully precomputed at `phase2_build`
and the depth loop ends up rebuilding tensors per depth in Python,
the redesign loses its forward-work advantage to per-depth
overhead.

Mitigation: enforce the "alloc-once, fill-in-place" contract in
`HybridPhase2Plan` (no GPU allocation in `phase2_build` or in the
depth loop).

### 3. Region isolation between continuation and proxy

Continuation rows write A_tail; proxy rows write B_proxy. If the
slot pool partitioning is wrong, one row can corrupt another's KV.

Mitigation: assert disjoint slot pools at `phase2_build`. The
hybrid-vs-split test also catches the symptom.

### 4. Graph capture fragility

8 graphs is a lot of capture surface. Each must capture cleanly at
its expected MQ_LEN.

Mitigation: capture-time errors are loud and one-time. v1
explicitly accepts the longer startup time.

---

## Non-Goals

- scheduler variable lookahead per case (v1 keeps max-case)
- B>1 MESA generalization
- EAGLE integration
- async SSD redesign outside MESA
- policy redesign beyond preserving current A/B behavior
