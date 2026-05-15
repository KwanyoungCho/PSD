# Phase C validation results

**Branch**: feat/mesa-proxy-async-overlap (worktree: feat/mesa-phase2-hybrid)
**Date**: 2026-05-15
**Verdict**: PASS — aligned plotter and schema-detect dispatch are working.

## Scope

Phase C delivers:

1. `ssd/bench/plot_mesa_aligned_timeline.py` — new aligned plotter that
   consumes the Phase-B JSON schema (rows with `_anchor` + `wall_start_ns`
   + `step_id` + `parent_label` + `status`) and produces
   `timeline_cache_hit_k1.png`, `timeline_cache_hit_k2.png`, and
   `timeline_cache_miss.png` joined by `step_id`.
2. `ssd/bench/summarize_ssd_run.py` — schema-detect dispatch:
   - Phase-B JSON  →  aligned plotter.
   - Legacy JSON   →  existing approximate plotter + one-line WARNING.
3. `phase_c/wait_shift_comparison.png` — visual confirmation of the
   `proxy_compute_send ↓` / `target_spec_wait ↑` shift across the
   engine-WIP boundary.

## Aligned plotter smoke checks

### 1. Phase-B 8B aligned trace, step 50

```
$ python bench/plot_mesa_aligned_timeline.py \
    experiments/proxy_async_overlap/phase_b/on --step-id 50 \
    --out /tmp/aligned_step50.png

by_status summary (target rank0):
  status      n_steps  wait_mean  wait_median   wait_p99
  hit_k1          252     13.585       13.449     14.495
  hit_k2            6     13.479       13.407     13.879
  miss             12     28.667       24.596     67.969
-> saved /tmp/aligned_step50.png
```

Result: PNG renders the two-row Gantt for step_id=50; target row shows
the `target_spec_wait_hit_k1` parent with the three handshake markers
(send/recv-wait/response-received) on top; draft row shows
`draft_recv_request → hit_cache_respond_hit_k1 → draft_send_response →
glue → 3× phase1 → phase2_build → ...`; a faint arrow connects
`draft_send_response.end` to `target_response_received.start` (the
~1.5 ms causal gap visible in the Phase-B JSON spot check).

### 2. Legacy JSON rejection

```
$ python bench/plot_mesa_aligned_timeline.py \
    experiments/paper_baselines/final_experiments/20260513_ours_k1_7_k2_7_*/ \
    --step-id 50 --out /tmp/should_fail.png

ERROR: target JSON in .../20260513_ours_k1_7_k2_7_... is not Phase-B aligned schema
(missing _anchor / wall_start_ns / step_id).
exit=2
```

Result: clean rejection with non-zero exit code as required.

### 3. Three-status PNGs in OUTDIR

```
$ python bench/plot_mesa_aligned_timeline.py experiments/proxy_async_overlap/phase_b/on

-> saved experiments/proxy_async_overlap/phase_b/on/timeline_cache_hit_k1.png
-> saved experiments/proxy_async_overlap/phase_b/on/timeline_cache_hit_k2.png
-> saved experiments/proxy_async_overlap/phase_b/on/timeline_cache_miss.png
```

Each PNG picks the median-`full_step_ms` step within its status bucket
(hit_k1: step 13/median 30.798 ms; hit_k2: step 40/median 30.412 ms;
miss: step 239/median 41.864 ms).

### 4. summarize_ssd_run.py schema-detect dispatch

Phase-B JSON path:

```
$ python bench/summarize_ssd_run.py experiments/proxy_async_overlap/phase_b/on --k 5

-> saved .../timeline_cache_hit_k1.png
-> saved .../timeline_cache_hit_k2.png
-> saved .../timeline_cache_miss.png
-> saved .../summary_metrics.csv
-> saved .../summary_metrics.md
```

Legacy JSON path:

```
$ python bench/summarize_ssd_run.py \
    experiments/paper_baselines/final_experiments/20260513_ours_k1_7_k2_7_*/ --k 14

[WARN] legacy MESA profile detected (no _anchor / wall_start_ns); timeline plots
use approximate handshake-offset alignment and should not be used as the paper
source of truth — re-run with the Phase-B aligned trace to get step-id-joined
timelines.
-> saved .../timeline_cache_hit.png
...
```

Result: WARNING is emitted; legacy approximate plotter runs unchanged.
Summary CSV still contains every metric the legacy run produced.

## Wait-shift comparison (doc 06 §6)

The current Phase-B 8B JSON cannot be compared apples-to-apples to the
70B K1=K2=7 baseline because the model size differs.  The only available
70B K1=K2=7 dumps post-engine-WIP are the legacy-schema `phase0b/A_detail0`
and `phase0b/B_detail1` runs.  Both predate the Phase-B `mesa_dump` rewrite,
so they do not carry `_anchor` / `wall_start_ns` / `step_id`.

Because both sides of the comparison are legacy schema, the wait-shift
figure is built from **per-event median cuda_ms** on each side, not from
the aligned timeline.  This is what doc 06 §6 already calls for: the
plotter is not required to render the comparison; the comparison only
needs to make the proxy ↓ / spec_wait ↑ pattern visually obvious.

Inputs (target rank 0, 70B K1=K2=7, warmup=50 events dropped per label):

| label                      | A: 2026-05-13 (pre-WIP) | B: phase0b A_detail0 (post-WIP) | delta |
|---                         |---:|---:|---:|
| `proxy_compute_send`       | 2.311 ms | 0.349 ms | **−1.962 ms** |
| `target_spec_wait_hit_k1`  | 4.124 ms | 6.714 ms | **+2.590 ms** |
| `graph_pre`                | 28.627 ms | 28.818 ms | +0.191 ms |
| `graph_post`               | 11.034 ms | 11.042 ms | +0.008 ms |

Pattern: `proxy_compute_send` drops by ~2.0 ms; `target_spec_wait_hit_k1`
grows by ~2.6 ms; `graph_pre` / `graph_post` are flat to within
measurement noise.  This matches the doc-06 §6 prediction and confirms
that the time visible in `proxy_compute_send` on the pre-WIP commit was
CUDA-stream / handshake waiting, not target compute — moving the wait
out of the proxy span (into `target_spec_wait_*`) is purely re-labelling
under the current target-side critical path.

Output: `wait_shift_comparison.png` (two-panel bar chart, generated by
`make_wait_shift.py`).

### Apples-to-apples 70B re-run

Skipped: no GPU access from this worktree.  A future 70B K1=K2=7
`PROFILE_MESA=1` re-run would produce a Phase-B-schema dump that the
aligned plotter can render directly; in that case
`wait_shift_comparison.png` can be regenerated by point-comparing the
two Phase-B JSONs (or by rendering the aligned timeline for the
representative miss step from each, side by side).

## Decisions where the spec was ambiguous

1. **Schema detection threshold.** Doc 06 §4.6 says the aligned plotter
   should *reject* legacy JSON.  The user-spec asks summarize_ssd_run.py
   to *fall back with a WARNING*.  The implementation reflects this
   exactly: the standalone CLI rejects (exit code 2); the
   summarize_ssd_run.py dispatch keeps the legacy plotter and prints a
   one-line WARNING.

2. **What counts as "aligned schema".** `is_aligned_schema()` accepts
   either the `_anchor` sentinel row *or* full `wall_start_ns` + `step_id`
   on every non-anchor row.  This avoids false negatives if anchor metadata
   is ever dropped while wall fields are present.

3. **Representative step selection.** Doc 06 §4.6 says "e.g. median of
   `full_step_ms`".  Implemented as median of the per-step span
   `max(wall_end_ns) - min(wall_start_ns)` over rows of that step,
   bucketed by canonical target_spec_wait status.

4. **Parent/child y-row layout.** Both the parent span (`target_spec_wait_*`,
   `phase2_build`) and its children share the same y; the parent is drawn
   first with alpha=0.35 (zorder=1) and children are drawn on top with
   alpha=0.95 (zorder=3).  This matches the doc-06 §4.6 wording
   ("translucent parent in background, opaque children on top") and avoids
   a separate phantom row for the handful of inner spans.

5. **Causality arrow.** Implemented only for `draft_send_response.end →
   target_response_received.start`.  Doc 06 mentions only this single
   handshake; no other arrows are drawn.

## Known limitations

- The aligned plotter assumes target rank 0 only (matches Phase-B trace
  scope).  Other TP ranks have no profile JSON to plot.
- If a draft profile has no rows for the chosen `step_id` (possible if
  the run was terminated mid-step), the draft row will be empty but the
  target row still renders; this is the intended graceful degradation.
- Warmup events: the plotter does not separately filter warmup steps.
  `pick_representative_step()` uses median across *all* steps with a
  given status, which in long runs is dominated by steady-state behavior.
  For very short runs (Phase-B smoke is 8 prompts × 128 out tokens) the
  hit_k2 / miss buckets are sparse (6 / 12 steps); the median is still
  a representative pick but the bucket has wide variance.
- The aligned plotter does not draw the legacy `verify_replay` /
  `verify_sample_accept` interleaving annotations from
  `_verify_replay_ms_by_wait_index`; that helper was specific to the
  legacy handshake-offset plotter's hit/miss step matching and has no
  analogue under step-id join.
- The `cpu_dispatch_start_ns` / `cpu_dispatch_end_ns` columns from the
  Phase-B schema are not rendered (doc 06 §4.3 calls them debug-only;
  CUDA-event `wall_start_ns` is authoritative for placement).

## Files added / modified

```
created   ssd/bench/plot_mesa_aligned_timeline.py
modified  ssd/bench/summarize_ssd_run.py
created   ssd/experiments/proxy_async_overlap/phase_c/RESULTS.md
created   ssd/experiments/proxy_async_overlap/phase_c/make_wait_shift.py
created   ssd/experiments/proxy_async_overlap/phase_c/wait_shift_comparison.png
created   (regenerated) ssd/experiments/proxy_async_overlap/phase_b/on/timeline_cache_hit_k1.png
created   (regenerated) ssd/experiments/proxy_async_overlap/phase_b/on/timeline_cache_hit_k2.png
created   (regenerated) ssd/experiments/proxy_async_overlap/phase_b/on/timeline_cache_miss.png
```

## Next steps

- Phase C-followup (deferred): re-run 70B K1=K2=7 with `PROFILE_MESA=1`
  to produce a Phase-B-schema dump on the post-WIP commit, then
  regenerate `wait_shift_comparison.png` from two Phase-B JSONs (or as
  two side-by-side aligned step PNGs).
- Phase D: archive `bench/plot_mesa_timeline.py` (the original
  approximate-anchor plotter) once the aligned plotter is the paper
  source of truth.  Do not delete the legacy plotter while
  `summarize_ssd_run.py` still uses it as the fallback path.
