# Batch 3v — 70B K1=K2=7 validation results

**Date**: 2026-07-02
**Setup**: 70B AWQ TP=4 + TinyLlama AWQ, K1=K2=7, exit=52, dfo=2, pfo=1,
ns=20 in=512 out=256, seed=42 temp=0.7, SSD_PROFILE_DUET=1 + DETAIL=1.

## Headline

| combo | ASYNC | STREAM | decode TPS | tok/step | accept | target step |
|---|---|---|---:|---:|---:|---:|
| off_off | 0 | 0 | 68.59 | 3.75 | 0.39 | 57.09 ms |
| on_off  | 1 | 0 | **74.36** | 4.05 | 0.44 | 57.14 ms |
| on_on   | 1 | 1 | 69.77 | 3.83 | 0.40 | 57.32 ms |

## Interpretation — the +8.4% is NOT attributable to the async send

The TPS difference decomposes entirely into tokens/step (3.75 → 4.05),
which follows accept_fraction (0.39 → 0.44). Target step time is flat
across all three combos (57.1 ms). The async send does not change what
tokens the proxy carries — only when the CPU unblocks — so a causal path
from isend to accept rate does not exist. With temp=0.7 sampling and a
timing-sensitive tree cache, accept_fraction has large run-to-run
variance; 0.39 vs 0.44 is within that variance band.

**Honest verdict: single-run noise. The async-send TPS gain is
UNCONFIRMED.** Repeat runs would be needed to resolve ±2-3 tok/s.

## Why the expected proxy_compute_send drop did not materialize

Target-side inner spans (mean ms):

| span | off_off | on_off | on_on |
|---|---:|---:|---:|
| proxy_compute | 0.860 | 0.858 | 1.092 |
| proxy_pack | 0.647 | 0.650 | 0.721 |
| proxy_send | **0.245** | 0.295 | 0.328 |
| proxy_compute_send (outer) | 1.889 | 1.947 | 2.356 |
| target_spec_wait | 6.887 | 6.725 | 6.287 |

In THESE runs the blocking send's peer-wait was already tiny
(`proxy_send` inner = 0.245 ms) — the draft posted its irecv early on
nearly every step, so there was almost nothing for isend to save. The
2.31 ms peer-wait seen in the 2026-05-18 breakdown occurs only when the
draft is deep in phase-1 replay when target reaches the send; that
condition did not dominate today.

on_on's outer grew (2.36 ms) because the CPU dispatch cost of the
stream switch (wait_event + record_stream ×4 + enqueue on a second
stream) is paid every step while the thing it hides (peer-wait) was
near zero. Batch 3b's overlap only pays off when peer-wait > dispatch
overhead (~0.4 ms).

## Ring stats

- `slot_wait_count` fired on 99.96% of sends — but `wait_ms_total` is
  only ~100 ms across ~5100 sends (**0.02 ms per fire**). With 2 slots
  and 1 send/step, the previous Work handle is always present when the
  slot rotates back, but the send completed long ago, so `wait()`
  returns instantly. The counter measures handle-reclaim frequency, not
  real stalls. (Interpretation note for summarize: use wait_ms, not
  wait_count.)

## Draft side — unchanged

proxy_wait 8.0 / 8.1 / 8.5 ms across combos; phase1/phase2 replay
identical. The draft pipeline is not affected by the target-side send
mode, as expected.

## Decision (doc 08 Phase 5 criteria)

- Correctness: PASS (byte-identical on 8B greedy; no corruption markers
  in 70B logs).
- Perf: UNCONFIRMED — the +8.4% on on_off is accept-rate noise, not a
  measured causal gain; on_on adds measurable dispatch overhead for no
  benefit at current peer-wait levels.

**Both gates stay default OFF.** The feature is retained as insurance
for configurations where the proxy peer-wait is large (e.g., higher
draft load, K1≫K2 shapes) — measure before enabling.

## Files

```
off_off/ on_off/ on_on/    run.log + duet_profile_*.json (gitignored) +
                           proxy_send_ring_stats.json
run.sh                     the 3-combo driver
```
