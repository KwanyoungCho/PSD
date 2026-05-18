# MESA Proxy Overlap — Implementation Report v2

## 0. Framing — this is a gated experiment, not a default-on optimization

Hypothesis: **MESA target-side proxy overhead (~2-3 ms per hit step) can be
hidden behind `graph_post` via a separate CUDA stream + async NCCL send,
because the proxy compute (~0.5 ms) + NCCL send wait (~1.5-2 ms) is much
smaller than `graph_post` (~13 ms).**

Expected outcome (realistic): **0% to +3% TPS**.
- Best case: +2-3% (proxy fully hides in graph_post)
- Worst case: 0% (wait-shifts back to target_spec_wait or graph_post slows
  due to SM contention with proxy compute)
- Failure modes: correctness break (record_stream miss); wait shift > saving

The experiment is structured to **prove or disprove the hypothesis cheaply**,
not to ship a confirmed win. Default behavior never changes unless Phase 5
explicitly flips defaults.

## 1. Architecture (corrected from v1)

### 1.1 What stays on default stream (all TP ranks)

```
default stream (all TP ranks, locked in sequence):
  graph_pre.replay()                              # layers [0..exit_layer]
  exit_h = exit_hidden + exit_residual            # small kernel
  normed = model.model.norm(exit_h, None)          # small kernel
  exit_logits = compute_logits(normed, last_only=False)
                                                   # ★ TP all-gather — MUST stay default
  ... (proxy callback dispatched, see below)
  graph_post.replay()                             # layers [exit_layer+1..L-1] + norm
  logits = compute_logits(outputs, last_only)     # TP all-gather
```

**Critical (reviewer #1)**: `compute_logits` performs an all-gather across
TP ranks. Moving it to `proxy_stream` would create asymmetric stream usage
across ranks (rank 0 on proxy_stream, ranks 1-3 on default stream) — a
recipe for subtle deadlocks or races. **Always default stream.**

### 1.2 What moves to proxy_stream (rank 0 only)

Only the **rank-0-only Policy B work** moves:

```
proxy_stream (rank 0 only):
  wait_event(exit_logits_ready)
  exit_logits.record_stream(proxy_stream)         # lifetime guard
  # ── inside mesa_proxy_fn / _compute_and_send_proxy:
  p_E = softmax(exit_logits[:, :K, :].float())     # ★ Policy B
  p_D = softmax(logits_q.float())
  ... gather / topk / cumprod / P_iv ...
  chosen_pos, chosen_tok = ...                    # rank-0-only
  ring.send(draft_rank, chosen_pos, chosen_tok)   # dist.isend (NCCL stream)
```

Ranks 1+ have `self._mesa_proxy_fn = None` and skip the callback entirely
([cudagraph_helpers.py:1344](ssd/ssd/engine/helpers/cudagraph_helpers.py#L1344)),
so there is no cross-rank stream asymmetry inside the proxy work.

### 1.3 Critical contract — default stream MUST NOT wait on proxy_stream

```python
# RIGHT — default stream continues immediately
ev_ready = torch.cuda.Event()
ev_ready.record()  # on default stream after exit_logits ready

with torch.cuda.stream(proxy_stream):
    proxy_stream.wait_event(ev_ready)
    mesa_proxy_fn(exit_logits, orig_bs)
# back to default stream — does NOT wait for proxy_stream

graph_post.replay()  # starts immediately on default stream
```

```python
# WRONG — defeats the purpose
... with stream(proxy_stream): mesa_proxy_fn(...)
torch.cuda.current_stream().wait_stream(proxy_stream)  # ★ DO NOT do this
graph_post.replay()
```

Reviewer #2: default stream waiting on proxy_stream would re-serialize.
The whole point is fire-and-forget; let proxy work race graph_post.

### 1.4 Passive overlap (NOT runtime-controlled timing)

The plan does **not** delay or schedule proxy work explicitly. It dispatches
proxy work to proxy_stream and lets the GPU scheduler interleave it with
graph_post on default stream. Success depends on:

```
T(proxy compute + NCCL send launch) ≤ T(graph_post) ≈ 13 ms
```

Currently `proxy compute + send` ≈ 0.5 + 1.5-2 = 2.0-2.5 ms ≪ 13 ms, so
the inequality holds **in principle**. SM contention may reduce effective
parallelism though — graph_post uses most SMs, proxy compute also wants
some SMs. This is the empirical question Phase 3 answers.

The phrase "slack-aware" used in earlier drafts is misleading — there is no
runtime control of slack. The validation gates (Phase 4) catch wait-shift
empirically; they do not prevent it.

## 2. Cross-stream lifetime checklist

| tensor | created on | read on proxy_stream | record_stream needed? |
|---|---|---|---|
| `exit_logits` | default (from compute_logits) | yes | **REQUIRED** — done at callback dispatch boundary in cudagraph_helpers.py |
| `draft_tokens` | default (from speculate_result) | yes (inside callback) | recommended (safety; verify_sample_accept also reads it later) |
| `logits_q` | default (from speculate_result) | yes (inside callback) | recommended |
| `cache_hits` | default | yes (inside callback) | recommended (None-guard before record_stream) |
| `chosen_pos` | proxy_stream (Policy B output) | proxy_stream only (ring copy) | NOT needed |
| `chosen_tok` | proxy_stream | proxy_stream only | NOT needed |
| ring slot buf | once at init (persistent) | proxy_stream (copy) + NCCL stream (send) | NOT needed (lifetime is process-long) |

**record_stream placement (reviewer #1)**: at the callback dispatch in
`cudagraph_helpers.py::run_mesa_verify_cudagraph`, right after
`exit_logits` is created and before entering the `with torch.cuda.stream(...)`
block. Doing it at the dispatch boundary makes "this tensor crosses streams"
visually explicit. Inside `_compute_and_send_proxy` we can add a defensive
re-call (harmless), but the boundary is the authoritative site.

## 3. AsyncSendRing — non-blocking proxy send

### 3.1 Data structure

```python
@dataclass
class AsyncSendSlot:
    buf: torch.Tensor              # persistent GPU int64 [wire_N * 2]
    work: object | None = None     # outstanding dist.Work (from isend)
    wait_count: int = 0            # how many times slot was reused while busy
    wait_ms_total: float = 0.0     # cumulative CPU wait at slot reuse


class AsyncSendRing:
    """N-slot round-robin for non-blocking proxy send.

    Each slot holds a persistent payload buffer + outstanding Work handle.
    Before reusing a slot, the previous send must complete (CPU-side wait).
    """
    def __init__(self, n_slots: int, buf_size: int, device, pg):
        self.slots = [
            AsyncSendSlot(
                buf=torch.empty(buf_size, dtype=torch.int64, device=device)
            ) for _ in range(n_slots)
        ]
        self.next_idx = 0
        self.pg = pg

    def send(self, dst: int, *tensors):
        slot = self.slots[self.next_idx]

        # Slot-reuse safety: rare path (n_slots tuned so this almost never fires)
        if slot.work is not None:
            import time
            t0 = time.perf_counter()
            slot.work.wait()                              # CPU blocks here
            slot.wait_ms_total += (time.perf_counter() - t0) * 1000
            slot.wait_count += 1
            slot.work = None

        # Copy payload into persistent buffer (no alloc on hot path).
        # copy_ runs on the current stream (proxy_stream when called from
        # within proxy callback).
        offset = 0
        for t in tensors:
            n = t.numel()
            slot.buf[offset:offset+n].copy_(t.view(-1), non_blocking=True)
            offset += n

        # Async send. dist.isend returns immediately with a Work handle.
        slot.work = dist.isend(slot.buf[:offset], dst=dst, group=self.pg)

        self.next_idx = (self.next_idx + 1) % len(self.slots)

    def drain(self):
        """Explicit cleanup. Called from engine shutdown path."""
        for slot in self.slots:
            if slot.work is not None:
                slot.work.wait()
                slot.work = None

    def dump_stats(self, outdir: str, decode_steps_seen: int):
        """Persist slot wait counters so Phase 4 / summarize can compute
        slot_wait_rate. Called by Verifier.drain_proxy_send_ring() under
        SSD_PROFILE_MESA=1 only — gate kept by caller."""
        import json
        out = {
            "slots": [
                {
                    "idx": i,
                    "wait_count": s.wait_count,
                    "wait_ms_total": s.wait_ms_total,
                }
                for i, s in enumerate(self.slots)
            ],
            "decode_steps_seen": decode_steps_seen,
        }
        with open(f"{outdir}/proxy_send_ring_stats.json", "w") as f:
            json.dump(out, f)
```

**Reviewer (v3 #1) — metric persistence path is mandatory**.
`AsyncSendRing.wait_count` / `wait_ms_total` live in memory; without
`dump_stats()`, Phase 4's `slot_wait_rate < 1%` gate is uncomputable.

Wiring:
```python
# Verifier.drain_proxy_send_ring():
if self._proxy_send_ring is not None:
    self._proxy_send_ring.drain()
    if os.environ.get("SSD_PROFILE_MESA", "0") == "1":
        outdir = os.environ.get("SSD_PROFILE_DIR", "/tmp")
        self._proxy_send_ring.dump_stats(
            outdir,
            decode_steps_seen=getattr(self, "_proxy_send_call_count", 0),
        )
```

Each `AsyncSendRing.send()` call also increments
`verifier._proxy_send_call_count` (denominator for the rate).

`summarize_ssd_run.py` reads `proxy_send_ring_stats.json` from the same
`SSD_PROFILE_DIR` as `mesa_profile_*.json` and emits
`prof_async_send_slot_wait_{count,rate,ms_avg}` columns.

**Reviewer #5 (CPU timer, not CUDA event)**: `work.wait()` is a CPU-side
blocking call; CPU `perf_counter` measures it correctly. CUDA events do
not measure CPU wait time and would add hot-path event overhead — the
v1 plan was wrong on this and is corrected here.

**Reviewer #3 (copy + isend ordering)**: `copy_` and `dist.isend` both go
through the current CUDA stream. PyTorch's NCCL backend handles stream
ordering internally (the isend kernel waits on prior stream work). This is
expected to be correct, but the Phase 2 stress test specifically watches for
token corruption that would indicate ordering issues. If we see corruption,
the fix is to record an event after `copy_` and have NCCL `wait_event` it
before send.

### 3.2 Sizing

- `wire_N` = `config.mesa_proxy_wire_N` ≈ 30-60 int64 (small)
- `buf_size = 2 * wire_N` ≈ 240-480 bytes per slot
- `n_slots = 2` initially; raised to 4 if Phase 2 measurement shows
  `slot_wait_count / decode_steps > 1%`

### 3.3 Cleanup — explicit drain, not `__del__`

```python
# Verifier:
def drain_proxy_send_ring(self):
    """Public cleanup hook. Call from llm_engine shutdown."""
    if self._proxy_send_ring is not None:
        self._proxy_send_ring.drain()

def __del__(self):
    """Backup only. Do NOT rely on this as primary cleanup."""
    try:
        self.drain_proxy_send_ring()
    except Exception:
        pass
```

```python
# llm_engine shutdown path (where the process group is still alive):
if hasattr(self.verifier, "drain_proxy_send_ring"):
    self.verifier.drain_proxy_send_ring()
self.draft_ps.join(...)  # only after drain
```

**Reviewer #6**: Python `__del__` is unreliable at interpreter exit. An
outstanding `dist.isend` interacting with a closed process group can hang
or segfault. Explicit drain from a known shutdown hook is required.

## 4. Env gates

| env | default | effect |
|---|---|---|
| `SSD_ASYNC_PROXY_SEND` | 0 | =1: ring buffer + `dist.isend` (Phase 2). Proxy compute still on default stream. |
| `SSD_PROXY_STREAM` | 0 | =1: Policy B compute + pack + send on proxy_stream (Phase 3). Requires `SSD_ASYNC_PROXY_SEND=1`. |

Both default OFF until Phase 5 explicit decision.

| ASYNC | STREAM | behavior |
|---|---|---|
| 0 | 0 | current production (blocking send, default stream) |
| 1 | 0 | non-blocking send, proxy compute still default stream → small saving expected |
| 0 | 1 | **invalid combo — fail-fast (assert) at engine init**; do NOT silently fallback |
| 1 | 1 | full proxy_stream overlap (target experiment) |

**Reviewer (v3 #4) — fail-fast on invalid combo**. Silent fallback would
hide a misconfiguration (e.g., user typed only `SSD_PROXY_STREAM=1` and
expected isend behavior). Validation at `Config.__post_init__` raises
`ValueError`:

```python
if os.environ.get("SSD_PROXY_STREAM", "0") == "1" \
   and os.environ.get("SSD_ASYNC_PROXY_SEND", "0") != "1":
    raise ValueError(
        "SSD_PROXY_STREAM=1 requires SSD_ASYNC_PROXY_SEND=1; "
        "Policy B compute on proxy_stream is meaningless without "
        "non-blocking send (the blocking send would re-serialize)."
    )
```

## 5. File-by-file changes

### 5.1 `ssd/utils/async_helpers/nccl_pack.py`
- Add `AsyncSendSlot` and `AsyncSendRing` classes (§3.1)
- Keep existing `send_int64` / `recv_int64` / `concat_int64` unchanged

### 5.2 `ssd/engine/verifier.py`
- `Verifier.__init__`: add `self._proxy_send_ring = None`
- `_compute_and_send_proxy`: env-gated dispatch:
  - `SSD_ASYNC_PROXY_SEND=1` → use `self._proxy_send_ring.send(...)`
  - Else → existing `send_int64`
- Add `drain_proxy_send_ring()` public method
- `__del__`: backup drain call

### 5.3 `ssd/engine/model_runner.py`
- `ModelRunner.__init__`: add `self._mesa_proxy_stream = None`
- `_ensure_proxy_stream()`: lazy init on rank 0 + MESA + `SSD_PROXY_STREAM=1`
  (called by Verifier when registering `_mesa_proxy_fn`)
- Config-time assertion: `SSD_PROXY_STREAM=1 requires SSD_ASYNC_PROXY_SEND=1`

### 5.4 `ssd/engine/helpers/cudagraph_helpers.py::run_mesa_verify_cudagraph`
- After `exit_logits = compute_logits(...)`:
  - If `SSD_PROXY_STREAM=1` and `model_runner._mesa_proxy_stream` is not None:
    - Record event on default stream
    - `exit_logits.record_stream(proxy_stream)` ← **at this boundary**
    - `with torch.cuda.stream(proxy_stream): proxy_stream.wait_event(ev); mesa_proxy_fn(...)`
    - Do NOT make default stream wait on proxy_stream
  - Else: existing direct call `mesa_proxy_fn(exit_logits, orig_bs)`
- `graph_post.replay()` immediately follows (default stream)

### 5.5 `ssd/engine/llm_engine.py`
- Shutdown path: `self.verifier.drain_proxy_send_ring()` before joining draft process

### 5.6 `ssd/bench/summarize_ssd_run.py`
- Parse and emit new fields from the profile:
  - `prof_async_send_slot_wait_count` (sum across slots)
  - `prof_async_send_slot_wait_rate` (= count / decode_steps)
  - `prof_async_send_slot_wait_ms_avg`
  - Per-status `target_spec_wait_{hit_k1,hit_k2,miss}_mean_ms`
  - `graph_post_ms` (already there, ensure mean reported)

## 6. Phase-by-phase plan

### Phase 0 — Clean baseline (no code changes, measurement only)

**Reviewer #4 + (v3 #3)**: TPS judgment must use `SSD_PROFILE_MESA=0`
(profile-off). Existing K1=K2=7 measurements (e.g., 70.81 tok/s from
breakdown) were PROFILE_MESA=1; not valid for performance comparison.

**Env hygiene — all of these must be unset / 0 for the clean baseline**:
```
SSD_PROFILE_MESA=0          ← off (no CUDA events / status calc)
SSD_PROFILE_MESA_DETAIL=0   ← off (no inner spans)
SSD_PROFILE_DRAFT (unset)   ← off
SSD_PROFILE_TARGET (unset)  ← off
SSD_PROFILE (unset)         ← off
SSD_TRACE_BUCKET (unset)    ← off
SSD_TRACE_SPLIT_K1K2 (unset)← off
```
Keep `SSD_FORCE_SPLIT_K1K2=1` (split-K1/K2 path under test).
Keep `SSD_PROFILE_DIR` for logging only (no profile JSON written when
`SSD_PROFILE_MESA=0`).

Run:
```bash
# 3 repeats, PROFILE_MESA=0 + all other profile envs off, current HEAD
for i in 1 2 3; do
  env -u SSD_PROFILE_DRAFT -u SSD_PROFILE_TARGET -u SSD_PROFILE \
      -u SSD_TRACE_BUCKET -u SSD_TRACE_SPLIT_K1K2 \
      SSD_PROFILE_MESA=0 SSD_PROFILE_MESA_DETAIL=0 SSD_FORCE_SPLIT_K1K2=1 \
      SSD_PROFILE_DIR=$PHASE_DIR/baseline_$i \
      python -O bench/bench.py <K1=K2=7 paper config> \
      > $PHASE_DIR/baseline_$i/run.log 2>&1
done
```

Output: 3-run decode_tps + target_full_step + accept_fraction.
Confirm baseline is stable (±1% across runs). Establish `TPS_baseline`.

**Why this matters for the gate**: Phase 5's `+1%` / `+2%` thresholds are
relative to this clean baseline. If the baseline itself was measured with
profiling on, comparison becomes noise-bound.

Commit: `feat(mesa): Phase 0 — clean PROFILE_MESA=0 baseline (3 reps)`

### Phase 1 — AsyncSendRing infrastructure (no behavior change)

Add `AsyncSendRing` / `AsyncSendSlot` to `nccl_pack.py`. Not wired in yet.
Add unit-ish test in `tests/` (CPU-only mock or NCCL roundtrip on 2-process).

Commit: `feat(nccl): AsyncSendRing class for non-blocking proxy send (unwired)`

### Phase 2 — `SSD_ASYNC_PROXY_SEND=1`

Wire in verifier.py. Proxy compute stays on default stream. Only the send
becomes non-blocking.

**Correctness gate** (must pass to proceed):
```bash
# Greedy temp=0, 3 seeds × 2 ASYNC env values
for SEED in 42 1337 9999; do
  for ASYNC in 0 1; do
    SSD_ASYNC_PROXY_SEND=$ASYNC SSD_FORCE_SPLIT_K1K2=1 \
    python -O bench/bench.py <K1=K2=7> --temp 0 --seed $SEED \
      --numseqs 5 --output_len 128 ...
  done
  # Generated tokens for ASYNC=0 vs ASYNC=1 must be byte-identical
done
```

**Perf measurement (PROFILE_MESA=0, 3 reps each)**:
- A: ASYNC=0 (baseline) — reuse Phase 0 numbers
- B: ASYNC=1

**Reviewer's revised decision criteria**:
- Correctness fail → STOP (Phase 3 not attempted)
- TPS negative but correctness OK → **proceed to Phase 3** (the heavier
  lever may still help; small Phase-2-only effect is expected because
  proxy compute is still default-stream-bound)
- TPS positive → proceed to Phase 3 to measure additional gain

Commit (B): `feat(mesa): SSD_ASYNC_PROXY_SEND gate — non-blocking proxy send via ring buffer`
Commit (perf): `chore(mesa): Phase 2 perf A/B (profile off)`

### Phase 3 — `SSD_PROXY_STREAM=1`

Move Policy B compute + pack + send to `proxy_stream`. `exit_logits` stays
on default stream (TP collective).

**Stress correctness** (record_stream miss detector):
```bash
# 200-step greedy, 3 seeds, byte-identical check
for SEED in 42 1337 9999; do
  SSD_ASYNC_PROXY_SEND=1 SSD_PROXY_STREAM=1 SSD_FORCE_SPLIT_K1K2=1 \
  python -O bench/bench.py <K1=K2=7> --temp 0 --seed $SEED \
    --numseqs 30 --output_len 256 ...
  # diff against Phase 0 baseline output (same seed, ASYNC/STREAM=0)
done
```

If even one token differs across runs, halt and check record_stream usage
on the 4 cross-stream tensors.

**Perf measurement**: A vs B vs C
- A: ASYNC=0, STREAM=0 (Phase 0 baseline)
- B: ASYNC=1, STREAM=0 (Phase 2)
- C: ASYNC=1, STREAM=1 (Phase 3)

3 reps each, PROFILE_MESA=0. Compare C to A.

Commit (B): `feat(mesa): SSD_PROXY_STREAM gate — Policy B on proxy_stream (exit_logits stays default)`
Commit (perf): `chore(mesa): Phase 3 perf A/B/C`

### Phase 4 — Wait-shift / per-status analysis (PROFILE_MESA=1)

Re-run A/B/C with `SSD_PROFILE_MESA=1` (3 reps each). Goal is **not**
performance numbers (profile overhead skews TPS); goal is to understand
where time moved.

**Per-status metrics**:
```
target_spec_wait_hit_k1   delta C vs A
target_spec_wait_hit_k2   delta C vs A
target_spec_wait_miss     delta C vs A
graph_post                delta C vs A    ← SM contention detector
draft_proxy_wait          delta C vs A
slot_wait_count / decode_steps             ← ring slot pressure
slot_wait_ms_avg                            ← fire-time wait
```

Generate Phase-C aligned timeline PNGs for representative steps under
condition C. Visually inspect whether proxy work overlaps graph_post or
slips into target_spec_wait / next-step territory.

Commit: `chore(mesa): Phase 4 wait-shift + per-status analysis (PROFILE_MESA=1)`

### Phase 5 — Default-on decision

**Decision table** (reviewer's revised criteria):

| Phase 3 result | Phase 4 wait-shift | Action |
|---|---|---|
| TPS Δ ≥ +2% (3-run avg, PROFILE_MESA=0) | All status gates pass | **Consider default-on**; do one more confirmation run |
| TPS Δ +1-2% | All gates pass | **Opt-in only**; keep env gate, document |
| TPS Δ < +1% | — | **Revert or gate-only**; document negative result in §6 of this doc |
| TPS Δ negative | — | Revert |
| Correctness fail at any point | — | Immediate revert |

**Per-status target_spec_wait gates** (each must hold):
- `hit_k1 Δ < 1.0 ms`
- `hit_k2 Δ < 1.0 ms`
- `miss   Δ < 1.0 ms`
- `graph_post_ms Δ < 0.5 ms` — **SM contention check, profile-on only** (v3 #5)
- `draft_proxy_wait Δ < 1.0 ms`
- `slot_wait_rate < 1%`

**Two-track measurement (reviewer v3 #3 + #5)**:

| metric | source run | reason |
|---|---|---|
| `decode_tps` (Δ %) | PROFILE_MESA=0 A/B/C | TPS judgment — profile overhead biases |
| `target_full_step_ms` | PROFILE_MESA=0 A/B/C | same reason |
| `target_spec_wait_{hit_k1,k2,miss}_mean_ms` (Δ ms) | PROFILE_MESA=1 A/B/C | profile required for status split |
| `graph_post_ms` (Δ ms) | PROFILE_MESA=1 A/B/C | profile required for span isolation |
| `draft_proxy_wait_ms` | PROFILE_MESA=1 A/B/C | profile required |
| `slot_wait_rate` / `slot_wait_ms_avg` | PROFILE_MESA=1 A/B/C | from `proxy_send_ring_stats.json` |

Both runs use the SAME `<K1=K2=7 paper config>`, only the profile envs
differ. Each PROFILE_MESA=1 run produces both `mesa_profile_*.json`
(target/draft) AND `proxy_send_ring_stats.json` (Phase 2/3 only) in the
same `SSD_PROFILE_DIR`.

Commit (default-on case): `chore(mesa): flip defaults — proxy async send + stream on (3-run +X.Y% confirmed)`
Commit (revert case): `chore(mesa): revert proxy overlap default; gate-only escape hatch retained`
Commit (negative result): `docs(mesa): Phase 5 result — proxy overlap hypothesis not confirmed (+X.Y% range)`

## 7. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| `record_stream` miss on one of 4 tensors → silent corruption | medium | high | Phase 3 stress (200 step × 3 seed); checklist |
| copy_ → isend ordering on NCCL backend | low | high | stress test catches; fallback to event-based sync |
| TP collective deadlock (compute_logits on wrong stream) | low | catastrophic | Plan keeps compute_logits on default stream (reviewer #1) |
| SM contention (proxy compute slows graph_post) | medium | low (-0.5-1 ms) | `graph_post_ms Δ < 0.5 ms` gate (reviewer #8) |
| Wait-shift (proxy_send ↓ but spec_wait ↑) | medium | medium (-saving) | Per-status spec_wait gates; PROFILE_MESA=0 TPS check |
| Slot exhaustion (ring 2 slots insufficient) | low | medium (CPU stalls) | `slot_wait_rate` gate; bump n_slots to 4 if hit |
| Outstanding isend at shutdown → hang/segfault | medium | high | Explicit `drain_proxy_send_ring()` in shutdown (reviewer #6) |
| Default-on with marginal gain → unstable benchmarks | medium | medium | Require +2% (3-run avg) for default-on (reviewer #10) |

## 8. Realistic expectation (reviewer #8)

> "성공해도 +1~3% 범위. graph_post와 proxy compute가 GPU 자원 경쟁하면 0%에 가까울 수도."

Confirmed. This experiment is **not** about shipping +X% TPS; it is about
**testing the hypothesis that target-side proxy overhead is hideable**.

Outcomes (any of these is informative):

1. **+2-3% confirmed**: hypothesis correct, default-on after triple check.
2. **+0-2%**: hypothesis partially correct (some hide, some SM contention).
   Keep gate-only as opt-in, document trade-offs.
3. **0% or negative**: hypothesis wrong — proxy overhead is not just NCCL
   wait, but SM-bound compute that re-emerges as graph_post slowdown or
   wait-shift. Revert; document; **token-side levers become the only path
   (Phase 2 hit quality)**.

All three results inform the next architectural decision. The current
PR scope is the **experiment**, not the claim.

## 9. Phase summary (one-shot reference)

```
Phase 0   Clean baseline (PROFILE_MESA=0, 3 reps)
            → establishes TPS_baseline
Phase 1   AsyncSendRing class (no wiring)
            → infra commit
Phase 2   SSD_ASYNC_PROXY_SEND=1 (non-blocking send only)
            → correctness gate (mandatory) + perf A/B
            → on correctness pass, proceed to Phase 3 regardless of TPS sign
Phase 3   SSD_PROXY_STREAM=1 (Policy B on proxy_stream, exit_logits stays default)
            → 200-step stress correctness + perf A/B/C
Phase 4   PROFILE_MESA=1 wait-shift / per-status / graph_post analysis
            → diagnostic, not performance
Phase 5   Decision table → default-on / opt-in / revert
            → final commit
```

## 10. What does NOT change in this experiment

- TP collectives all stay on default stream (compute_logits, graph_pre / graph_post replays' internal collectives)
- Policy B algorithm unchanged
- Cache key structure unchanged
- mesa_phase1_k / mesa_phase2_k / dfo / pfo unchanged
- Phase 2 hit quality (L_p2 = 2.21 ms) is NOT addressed here — that is a
  separate algorithmic track that the current experiment does not affect
- Hybrid path / non-split path NOT touched (env gates are split-K1/K2 specific)

## Appendix A. Why this is a separate doc from `06-timeline-cleanup-plan.md`

`06-` is the trace/measurement infrastructure (Phase B + C of that track,
already merged). `07-` (this doc) is the first experiment that **uses**
that infrastructure to test a specific optimization hypothesis. The
aligned trace + status-labeled spans from `06-` are what makes the
Phase 4 wait-shift analysis precise enough to draw conclusions.
