# DUET timeline / profiling cleanup plan

## 0. Executive summary

The current bottleneck analysis should stop treating `proxy_compute_send` as the
main optimization target. Older runs correctly showed a long
`proxy_compute_send` parent span, but that span included synchronization /
stream-visible waiting. In the current WIP, that visible wait moved mostly into
`target_spec_wait`, while decode TPS did not improve.

For target-GPU execution overhead, the direct DUET-added target work is small:
`exit_logits` plus `proxy_compute_send` is roughly sub-ms to ~1 ms per target
step in the current measurements. The larger variability comes from pipeline
waiting: target finishes its own work and waits for draft-side response.

The next goal is not another proxy-send optimization. The next goal is a
trustworthy timeline that answers:

> When target is executing event X in step N, what is draft doing in the same
> physical time interval?

The current timeline plots do not reliably answer that question because target
and draft profiles use process-local CUDA event origins and are aligned by an
approximate handshake anchor.

## 1. Current state

### 1.1 What is reliable today

- Per-process CUDA event durations are useful.
- Target-only ordering is reliable.
- Draft-only ordering is reliable.
- Aggregate metrics such as decode TPS, target full step, target verify,
  target spec wait, draft step, P1/P2 hit rate, and accepted length are useful.

### 1.2 What is not reliable enough today

- Absolute target/draft alignment in `timeline_cache_*.png`.
- Interpreting a long parent span such as `proxy_compute_send` as pure compute.
- Comparing old timeline PNGs across commits as if they represent the same
  critical path.

The old `timeline_cache_hit_k1.png` from `20260512_ours_label_perf...` is not
fake: that run really recorded a ~2.2 ms target `proxy_compute_send` span.
However, it is a parent span and includes waiting. It should not be used as the
current bottleneck explanation after later WIP commits moved that wait elsewhere.

## 2. Metric definitions

Use throughput-oriented names in reports.

| Metric | Meaning | Use |
|---|---|---|
| `decode_tps` | generated decode tokens / decode wall time | primary performance metric |
| `target_full_step_ms` | target-side per-step critical path | target GPU timeline cost |
| `target_verify_ms` | target verify path excluding some waits/post work | verify cost reference |
| `target_spec_wait_ms` | target idle/pipeline wait for draft response | main wait-shift detector |
| `proxy_compute_send_ms` | target parent span between exit logits and graph_post | direct target proxy region, may include wait |
| `proxy_inner_sum_ms` | DETAIL-only compute + pack + send | diagnostic only |
| `proxy_unattributed_stall_ms` | parent minus inner sum | diagnostic only |
| `draft_proxy_wait_ms` | draft waiting for target proxy payload | DUET pipeline wait |
| `p1_hit`, `p2_hit` | cache hit source split | algorithm behavior |
| `p1_avg_accepted_len`, `p2_avg_accepted_len` | accepted speculative length on each hit source | algorithm quality |

Do not use pure inner breakdown alone to claim throughput improvement. A
decrease in one target label is only real if `decode_tps` improves and the time
does not reappear in `target_spec_wait` or draft-side waits.

## 3. Keep / remove policy

### Keep for real experiments

- `SSD_PROFILE_DUET=1`
  - Main profiling switch for paper-quality DUET timeline/summary.
  - It should be the only profiling env needed in normal performance runs.

### Keep only as temporary debug tools

- `SSD_PROFILE_DUET_DETAIL=1`
  - Use only for one-off proxy inner breakdown.
  - Do not enable in baseline or paper comparison runs.
- `SSD_TRACE_SPLIT_K1K2=1`
  - Contract debugging only.
- `SSD_TRACE_BUCKET=1`
  - Bucket dispatch debugging only.
- `SSD_PROFILE_DRAFT=1`, `SSD_PROFILE_TARGET=1`, `SSD_PROFILE=1`
  - Development/debug profiling only. These can add syncs or produce
    non-paper metrics.

### Remove / avoid

- Do not add new profiling env vars for each experiment.
- Do not keep verbose pre-dump prints in hot or shutdown paths.
- Do not use old standalone timeline scripts as the paper source of truth once
  the aligned timeline path below exists.

Cleanup already applied:

- Removed `SSD_DRAFT_EXIT_TIMEOUT` as a user-facing env knob. Profiling runs now
  get a longer draft shutdown grace period automatically when
  `SSD_PROFILE_DUET=1`.
- Removed draft profile pre-dump debug print / traceback spam. A concise warning
  remains only if profile dumping fails while `SSD_PROFILE_DUET=1`.

## 4. Required new timeline design

### 4.1 Problem

CUDA event timestamps are process-local. Target and draft events each have a
valid internal time axis, but there is no exact shared zero point. The current
plot aligns them by a handshake heuristic such as:

```text
target_spec_wait.end ≈ draft_send_response.end
```

This is useful for intuition but not strong enough for final analysis. Blocking
send/recv, CUDA stream order, and CPU scheduling can move visible wait between
labels.

### 4.2 Time base: CUDA-event anchored wall clock

Do not use CPU timestamps alone as the cross-process alignment axis. CPU
timestamps measure Python dispatch/enqueue time, not the time at which queued
CUDA work reaches the GPU stream. A target stream may have tens of milliseconds
of queued graph work, so `time.perf_counter_ns()` taken at `duet_record()` can
precede the actual GPU event by the whole queue depth.

Instead, initialize one CUDA/CPU anchor per process when `SSD_PROFILE_DUET=1`:

```text
torch.cuda.synchronize()
anchor_cpu_ns = time.perf_counter_ns()
anchor_event = torch.cuda.Event(enable_timing=True)
anchor_event.record()
torch.cuda.synchronize()
```

At each `duet_record()` / `duet_close()`:

```text
start_event.record()                 # GPU-stream timestamp
start_cpu_dispatch_ns = perf_counter_ns()  # debug/reference only
...
end_event.record()
end_cpu_dispatch_ns = perf_counter_ns()
```

At dump time, convert GPU event times to a shared monotonic-clock estimate:

```text
gpu_start_ms_since_anchor = anchor_event.elapsed_time(start_event)
gpu_end_ms_since_anchor   = anchor_event.elapsed_time(end_event)
wall_start_ns             = anchor_cpu_ns + int(gpu_start_ms_since_anchor * 1e6)
wall_end_ns               = anchor_cpu_ns + int(gpu_end_ms_since_anchor * 1e6)
cuda_ms                   = start_event.elapsed_time(end_event)
```

This makes placement use GPU execution time, while still expressing both
processes on the same host monotonic clock. The only synchronizations are at
profile initialization and dump, not in the per-step path.

Dump process metadata once:

```text
proc
anchor_cpu_ns
anchor_device
anchor_note = "GPU event times converted to host monotonic clock"
```

### 4.3 New trace row schema

Extend the existing `duet_record` / `duet_close` path under `SSD_PROFILE_DUET=1`
to write CUDA-anchored placement fields plus CPU dispatch fields.

Required row fields:

```text
idx
proc                    # target_rank0 or draft
step_id                 # monotonically increasing speculative step id
status                  # hit_k1, hit_k2, miss, mixed, or unknown
label
parent_label            # optional; null for root spans
gpu_start_ms_since_anchor
gpu_end_ms_since_anchor
wall_start_ns
wall_end_ns
cuda_ms                  # duration from CUDA events
cpu_dispatch_start_ns    # debug only: Python enqueue time
cpu_dispatch_end_ns      # debug only: Python enqueue time
```

Notes:

- CUDA event duration remains the authoritative per-event GPU duration.
- `wall_start_ns` / `wall_end_ns` provide the cross-process placement axis.
- CPU dispatch timestamps are not used for final alignment. They are included
  only to debug enqueue lag or CPU scheduling artifacts.
- `step_id` is mandatory. Do not rely on event index matching.

### 4.4 Step context and wire-through

Avoid changing every call site. Add a tiny profiler context in
`cudagraph_helpers.py`:

```text
duet_set_context(step_id=None, status=None, proc=None)
duet_record(label) reads the current context
duet_close(label, start_event) writes context into the row
```

Target and draft must agree on the same `step_id`. Do not infer it from event
index order. Wire it through the async request:

1. Add `request_step_id` to `SpeculatorAsync`.
2. Increment it once per `SpeculatorAsync._speculation_request()` call.
   It is a monotonically increasing int64 and must not reset or be reused within
   one target process lifetime.
3. Send it in the spec request metadata. Current spec metadata is
   `[B, K, F]`; change it to `[B, K, F, step_id]`.
4. In `DraftRunner._service_spec_request()`, receive metadata shape `(4,)` and
   set the draft profiler context to that `step_id`.
5. Attach `step_id` to `SpeculateResult` so `SpecDecodeStep.decode()` and
   `Verifier.verify()` can set target-side context for target verify events.

Semantics:

- `step_id` names one target request/response cycle.
- Target `target_spec_wait_*` belongs to the request whose response is being
  received.
- Target verify events after that response use the same `step_id`.
- Draft cache/tree work spawned after `draft_send_response` also keeps that
  request's `step_id`, because it explains what the draft was doing while target
  verified the response.

Labels inside the step inherit the current context unless an inner scope
explicitly overrides status.

`duet_close()` reads the current context at close time, not only the context at
the matching open. This is required for spans such as `target_spec_wait_*`,
where the hit/miss status is only known after `speculator.speculate()` returns.
The context is process-local module state; this is sufficient because target
rank 0 and draft rank are single-threaded for the profiled path.

### 4.5 Handshake markers

Add explicit low-overhead markers for alignment and causality:

| Marker | Process | Parent | Call site | Meaning |
|---|---|---|---|---|
| `target_send_request` | target | `target_spec_wait_*` | `SpeculatorAsync._speculation_request()`, from before `dist.send(self._cmd)` through the fused request payload send and optional EAGLE payload sends | target sends one spec request to draft |
| `target_recv_response_wait` | target | `target_spec_wait_*` | `SpeculatorAsync._speculation_request()`, around `dist.recv(self._fused_response)` and `dist.recv(self._logits_q)` | target waits for draft response payload |
| `target_response_received` | target | `target_spec_wait_*` | immediately after logits recv completes | point marker that the response is available |
| `target_spec_wait_*` | target | null | existing parent in `SpecDecodeStep.decode()` around `speculator.speculate()` | throughput-visible target idle/request/response parent span |
| `draft_recv_cmd` | draft | null | existing marker in `DraftRunner.draft_loop()` around `self.recv_cmd()` | draft waits for next command |
| `draft_recv_request` | draft | null | `DraftRunner._service_spec_request()`, from metadata recv through fused request payload recv and optional EAGLE payload recv | draft receives request payload |
| `hit_cache_respond_*` | draft | null | existing marker around `hit_cache_and_respond()` | cache hit/miss classification and immediate response construction |
| `draft_send_response` | draft | null | existing marker around response token/logit sends | draft sends response to target |
| `proxy_wait` | draft | phase2 build span | existing DUET marker inside phase-2 build | draft waits for target proxy payload |

Existing labels can remain, but the plotter must use these markers to explain:

- which request each target wait is associated with,
- which draft response unblocked that wait,
- what draft work overlapped target `graph_pre`, `graph_post`, and
  `verify_sample_accept`.

### 4.6 New plotter behavior

Create a new plotter, or replace `summarize_ssd_run.py` timeline generation
after validation:

```text
bench/plot_duet_aligned_timeline.py OUTDIR --step-id N [--causality-shift]
```

Rules:

- Join target and draft by `step_id`, not by global event index.
- Use `wall_start_ns` / `wall_end_ns` for cross-process placement.
- Use CUDA `cuda_ms` for duration labels.
- Draw `target_spec_wait` as idle wait and annotate the draft event it waits for.
- Do not hide parent spans silently. If DETAIL inner spans exist, draw parent as
  a translucent background and inner spans on top.
- Trace only `target_rank0` for target TP runs. Rank 0 is the timeline source of
  truth because it owns the DUET proxy/async handshake path; other TP ranks are
  synchronized through collectives and do not need separate paper plots.

### 4.7 Cross-process clock drift — known limitation + `--causality-shift`

Each process anchors its own CUDA-event timer at startup (§4.2).  Target rank 0
and the draft process run on *different physical GPUs* whose hardware clocks
differ by ~1-10 ppm.  Over a long run this accumulates ms-level skew in the
wall-clock conversion:

  - per-event GPU duration (`cuda_ms`): unaffected (single-stream measurement)
  - within-process ordering: unaffected
  - cross-process placement (`wall_start_ns` differential): drifts

Symptom: at step_id N late in a long run, `draft_send_response.wall_end_ns >
target_response_received.wall_start_ns` even though the response physically
arrived at target before target unblocked — a causality violation in the
displayed wall_ns.  Observed magnitude: e.g. +5.7 ms after ~830 s (≈6.8 ppm
drift, well within normal GPU clock tolerance).

Mitigation: `plot_duet_aligned_timeline.py --causality-shift` shifts the draft
row by the median `draft_send_response.end − target_response_received.start`
offset over ±`--causality-window` steps around the rendered step (default
window=5, so 11 response pairs).  This is a **visual-only correction**:

  - JSON data is not modified.
  - The reference causal pair is `draft_send_response.end →
    target_response_received.start`, NOT `target_send_request.end →
    draft_recv_request.start` — `draft_recv_request.start` is when draft *posts*
    the recv, which may precede target's send when draft has the irecv pre-posted.
    Only the response pair imposes a strict wall-time ordering.
  - The title carries `[draft causality-shifted ±X.XXX ms (median over N pairs)]`.
  - Median over ±N steps rejects single-pair noise (NCCL handshake jitter,
    one-off pauses).

When to enable:

  - Paper-figure renders on long runs (>5 minutes wall time).
  - Any step where the raw plot shows draft starting visibly later than the
    matching target send.

When NOT to enable:

  - Short profile runs (<5 min) where drift is sub-ms and uncorrected plots
    are already accurate.
  - When you want to diagnose the drift itself.

The exact-anchor alternative — periodic re-anchoring every N steps — is more
accurate but adds CUDA syncs to the per-step path and requires engine-side
changes.  Out of scope for the current paper figure goal; revisit only if
plotter-only correction proves insufficient.

## 5. Implementation plan

### Phase A — cleanup current WIP

Status: partially done.

- Remove extra shutdown env knobs.
- Remove verbose draft pre-dump debug prints.
- Keep profile dump failure visible but concise.
- Do not touch unrelated experiment deletions or model/kernel changes.

### Phase B — add low-overhead aligned trace

Files:

- `ssd/ssd/engine/helpers/cudagraph_helpers.py`
- `ssd/ssd/engine/speculator_async.py`
- `ssd/ssd/engine/draft_runner.py`
- `ssd/ssd/engine/verifier.py`
- `ssd/ssd/engine/helpers/speculate_types.py`
- `ssd/ssd/engine/step.py`

Tasks:

- Add CUDA-event anchor metadata and wall-clock conversion to
  `duet_record` / `duet_close` / `duet_dump`.
- Add process-local profiler context with `step_id` and `status`.
- Wire `step_id` through async spec metadata: `[B, K, F]` → `[B, K, F, step_id]`.
- Add `step_id` to `SpeculateResult`.
- Set target context around `target_spec_wait`, target verify, and target
  postprocess for the matching `step_id`.
- Set draft context around each draft service step from the received `step_id`.
- Add handshake markers listed in Section 4.5.
- Ensure status labels are `hit_k1`, `hit_k2`, `miss`, `mixed`, or `unknown`.
- Keep all of this gated by `SSD_PROFILE_DUET=1`.

Performance rule:

- No `torch.cuda.synchronize()` in the step path.
- No `.item()` solely for profiling in the step path.
- No per-event file I/O.
- No new env var unless there is no reasonable alternative.
- The only allowed synchronizations for the aligned timeline are profile anchor
  initialization and final dump.

### Phase C — replace timeline generation

Files:

- `ssd/bench/summarize_ssd_run.py`
- new optional helper: `ssd/bench/plot_duet_aligned_timeline.py`

Tasks:

- Generate `timeline_cache_hit_k1.png`, `timeline_cache_hit_k2.png`, and
  `timeline_cache_miss.png` from step-id aligned rows.
- Add a warning to old timeline mode if rows do not contain `step_id` and
  `wall_start_ns`.
- Keep old plotting only for legacy profile files, clearly labeled as
  approximate.
- Re-plot the old wait-shift comparison if raw profiles are available:
  `20260513` pre-WIP baseline versus current WIP. The new plotter should make
  the movement from `proxy_compute_send` into `target_spec_wait` visible rather
  than hiding it.

### Phase D — remove obsolete debug paths

After Phase B/C are validated:

- Stop using `bench/plot_duet_timeline.py` for paper figures.
- Archive stale experiment scripts only from an explicit list. Do not delete or
  move `ssd/experiments/paper_baselines/final_experiments`,
  `ssd/experiments/paper_baselines/ssd_dense_7b_amd135m_split`, or any directory
  containing final paper tables/PNGs unless the user explicitly approves the
  path list.
- Keep `SSD_PROFILE_DUET_DETAIL` only if there is still an active use case for
  proxy inner breakdown. Otherwise remove the env and the inner spans.

## 6. Validation

### Correctness

- Run greedy temp=0 twice with `SSD_PROFILE_DUET=0` and `SSD_PROFILE_DUET=1`.
- Generated tokens should match exactly.
- P1/P2 hit counts should match.

### Profile overhead

Compare the same run with profiling off/on:

```text
decode_tps delta <= 1%
avg_tokens_per_step delta <= noise
p1_hit / p2_hit delta <= noise
```

If profile-on changes throughput meaningfully, the trace is too heavy.

The additional CPU timestamp calls are expected to be negligible. The main
overhead remains CUDA event recording, which already exists in the current
`SSD_PROFILE_DUET` path.

### Timeline sanity

For a chosen `step_id`:

- Target `target_recv_response_wait` must overlap the draft work that produces
  that response.
- Hit/miss status in target wait must match draft cache response status.
- K1/K2 hit timelines must be separable.
- Parent/inner spans must not double-count in summary tables.
- The aligned plot must reproduce known wait-shift behavior: if a newer commit
  moves time from `proxy_compute_send` to `target_spec_wait`, the plot should
  show that movement instead of changing only label colors or anchoring.

## 7. Current recommendation

Do not proceed with async proxy send / proxy stream optimization yet. The current
data says `proxy_compute_send` is no longer the active bottleneck, and prior
changes already demonstrated wait-shift into `target_spec_wait`.

Proceed with:

1. cleanup of measurement WIP,
2. low-overhead aligned timeline trace,
3. `target_spec_wait` heavy-tail attribution by step status,
4. only then decide whether draft-side or target-side code optimization is worth
   doing.
