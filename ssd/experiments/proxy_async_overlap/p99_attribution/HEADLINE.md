# target_spec_wait p99 attribution — headline

**Run**: 70B AWQ, K1=K2=7, exit_layer=56, --temp 0.7 --seed 42 --numseqs 50
--input_len 512 --output_len 256 --all, SSD_FORCE_SPLIT_K1K2=1.
**Branch**: feat/mesa-proxy-async-overlap @ abe51d3 + Phase B aligned trace.
**Sample**: 13,427 spec steps (target rank0 161 124 spans / draft 510 228 spans).

## Bottom line

The `target_spec_wait p99 ≈ 19.96 ms` tail observed in earlier
runs is **NOT** dominated by handshake / NCCL / proxy work. It is
structurally the cost of **draft cache misses**, where `draft.hit_cache_and_respond`
falls through to `jit_speculate` and runs K1=7 forward passes from scratch
before responding.

| | n | mean | p50 | p90 | p99 | total share |
|---|---:|---:|---:|---:|---:|---:|
| hit_k1   | 7 524 | 4.46 ms | 4.31 | 5.54 | **6.21** | 33 542 ms (36.4 %) |
| hit_k2   | 3 201 | 4.37 ms | 4.16 | 5.46 | **6.24** | 13 978 ms (15.2 %) |
| **miss** | 2 702 | **16.55 ms** | 16.52 | 17.91 | **18.62** | **44 723 ms (48.5 %)** |
| overall  | 13 427 | 6.86 ms | 4.31 | 16.97 | 18.55 | 92 243 ms |

**Read this table as**: misses are 20 % of steps but ≈ 48.5 % of all wait
time. Per-step they cost 4× a hit. p99 is also entirely miss-driven (hit
p99 is only 6.2 ms).

## What draft is doing during the wait (top miss outliers)

Each of the top miss outliers (~18-22 ms wait) is overwhelmingly
`hit_cache_respond_miss` on the draft side:

| step_id | wait | `hit_cache_respond_miss` | second largest | unattributed |
|---:|---:|---:|---|---:|
| 904  | 17.9 ms | 13.43 ms (75 %) | `phase2_replay` prev | 0.4 % |
| 1014 | 18.6 ms | 13.37 ms (72 %) | `phase2_replay` prev | 0.2 % |
| 457  | 18.6 ms | 13.32 ms (71 %) | `phase2_replay` prev | 0.5 % |
| 576  | 18.6 ms | 13.27 ms (71 %) | `phase2_replay` prev | 0.3 % |

`hit_cache_respond_miss` is the draft path in
`DraftRunner.hit_cache_and_respond` where the request misses the tree
cache, so the function calls `jit_speculate(K=7)` — K1 sequential
draft-model forwards from the recovery token. 7 forwards × ~1.8 ms = ~12.6
ms exactly matches the observed share. There is no MESA bug here; this is
the work that has to happen on a miss.

## What about the 200-400 ms outliers?

| status | step_id | wait | draft overlap sum | unattributed |
|---|---:|---:|---:|---:|
| hit_k1 | 10579 | 389.7 ms | ~37 ms (~10 %) | **~90 %** |
| hit_k1 | 4923  | 300.8 ms | ~37 ms (~12 %) | **~88 %** |
| hit_k1 | 1282  | 232.4 ms | ~34 ms (~15 %) | **~85 %** |
| hit_k2 | 7427  | 335.6 ms | — | similar |
| hit_k2 | 2922  | 262.6 ms | — | similar |

These rare events (5 total out of 13 427 spec steps) account for ~1 % of
the cumulative target wait time. In each, draft has **no measured work
covering 85-90 % of target's wait window**, which points at system-level
pauses (GC, NCCL retry, kernel page faults, scheduler stalls) rather
than an algorithm-side bottleneck. Not worth optimizing as a class —
they're noise.

## What this means for the next optimization

The right framing is no longer "shave milliseconds off proxy_compute_send"
(Phase 0b proved that's exhausted) and no longer "the p99 tail of
target_spec_wait" as if it were one effect. It is two distinct things:

1. **Miss cost — the actual paper-relevant headroom.**
   - Reduce miss rate: improve cache key match / recovery-token
     prediction so fewer requests fall through to `jit_speculate`.
   - Reduce miss cost: parallelize / shorten `jit_speculate` (hard —
     it is autoregressive on the draft model).
   - Hide miss latency: pre-warm the cache earlier, overlap miss with
     target prefill or with the previous step's tail.
   - Headroom: if a 50 % miss reduction is feasible, that recovers
     ≈ 22 000 ms of wait over 13 427 steps → ≈ 1.7 ms/step ≈ 3 % TPS.
   - Headroom: if miss cost drops 30 % (e.g., a partial pre-fetch
     scheme), that recovers ≈ 13 000 ms → ≈ 1 ms/step ≈ 1.7 % TPS.

2. **Rare 200-400 ms outliers — NOT an algorithm problem.**
   - Sum to ~1 % of total wait time.
   - 85-90 % of each outlier window is unattributed (no draft work).
   - Likely system-level (NCCL, GC, scheduler). Investigate only if a
     specific workload shows persistent regression.

## Phase 0b conclusion confirmed

The aligned timeline reproduces the earlier finding: `proxy_compute_send`
and `target_spec_wait` are interchangeable cost slots from target's
point of view. The actual binding constraint for end-to-end throughput
on this configuration is the **draft jit_speculate path on misses**, not
the target rank 0 proxy send path. Moving wait between target labels
(as the engine WIP between 2026-05-13 and Phase B did) does not improve
TPS because the underlying serial work on the draft side is the limit.

## Artifacts

```
p99_attribution/
  sanity/
    run_sanity.sh                                     # overhead gate script
    {off,on}/run.log                                   # PROFILE_MESA=0/1 logs
    on/mesa_profile_*.json                             # Phase-B JSON from sanity
  full/
    run_full.sh                                        # full attribution run
    run.log                                            # bench output
    mesa_profile_target_rank0_184434.json   (161 k spans)
    mesa_profile_draft_184439.json          (510 k spans)
    RESULTS.md                                          # per-step attribution tables
    attribution.json                                    # machine-readable summary
    timeline_p99_hit_k1_step10579.png                  # 389.7 ms outlier
    timeline_p99_hit_k2_step7427.png                   # 335.6 ms outlier
    timeline_p99_miss_step1.png                        # 84.0 ms warmup outlier
  analyze_p99.py                                       # the analyzer
  HEADLINE.md                                          # this file
```

## Followups (not done in this commit)

- Repeat with a different seed to confirm the miss-share is stable
  across draws.
- Compare hit_cache_respond_miss vs hit_cache_respond_hit_k1 durations
  directly (draft side) to quantify the cache-key effectiveness gap.
- Look at whether miss steps cluster temporally (warmup early-context)
  or spread evenly; if early, more aggressive pre-warm helps; if
  uniform, structural cache work is needed.
