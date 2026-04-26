# MESA Phase 2 Hybrid — Implementation Issue Tracker

This file tracks issues discovered during implementation of the
Phase 2 hybrid redesign (per `MESA-PHASE2-HYBRID-IMPLEMENTATION-PLAN.md`).
Each issue: how it surfaced, root cause, resolution.

---

## Implementation Strategy Notes (2026-04-26)

### Phase 3b sub-staging

The plan's Phase 3 ("Phase 1 K1 split + Hybrid Phase 2 long-base bucket
only") is internally a multi-piece change:

1. Phase 1 forward depth shortening (`K_long` → `K1`)
2. Phase 2 continuation (extending Phase 1 leaves by `K2` more depths)
3. Phase 2 proxy depth shortening (`K_long` → `K2`)
4. Combining continuation + proxy into a single hybrid forward
5. Per-row attention plumbing (custom block tables, mask builder)
6. Long-base CUDAGraph capture for the hybrid path

To keep each commit reviewable and runtime-debuggable, Phase 3b is
implemented in three sub-stages:

- **3b.1 — Phase 1 K1 split** (this commit): create
  `phase1_layout_long` with `forward_depth = K1`. Existing
  `_build_tree_batch_mesa` runs Phase 1 with K1 depth, then a NEW
  Phase 2 continuation pass extends each Phase 1 leaf by `K2` more
  depths. Phase 2 proxy keeps `K_long` for now. Total layouts run
  per step: 3 (Phase 1, Phase 2 continuation, Phase 2 proxy). Cache
  emit shape unchanged (draft-sourced still `K_long`).
- **3b.2 — Phase 2 proxy at K2** (next): shorten Phase 2 proxy
  forward depth to `K2`. Proxy-sourced rows now have suffix length
  `K_short = K2`. Cache emit gets two row classes.
- **3b.3 — Single hybrid forward** (final): merge Phase 2
  continuation + proxy into a single batched forward. This is the
  step that requires the per-row attention plumbing and custom mask
  builder. Most engineering risk lives here.

This staged approach preserves end-to-end MESA correctness at each
sub-stage and lands the forward-work reduction in 3b.1 (Phase 1 going
from K_long → K1 saves the largest chunk).

### Runtime-validation gating

CUDAGraph capture changes in Phase 3 / 5 cannot be validated by static
analysis alone — they require a real GPU run with a 70B AWQ + draft
AWQ stack to surface FlashInfer wrapper / mask shape mismatches.
Phase 6 (validation) is the gate; if a sub-stage fails its smoke run,
the issue is documented here and a fix lands before moving on.

---
