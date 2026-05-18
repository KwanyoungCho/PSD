# Phase 0 — Clean baseline (PROFILE_MESA=0)

**Branch**: feat/mesa-proxy-async-overlap @ 79e06a0
**Date**: 2026-05-18
**Goal**: establish TPS_baseline against which Phase 2/3/5 gates are measured.

## Config (same as 20260513 paper baseline)

```
K1=K2=7, dfo=2, pfo=1, exit=52, mesa_policy=b, SSD_FORCE_SPLIT_K1K2=1
70B AWQ TP=4 target + TinyLlama-1.1B AWQ TP=1 draft
--temp 0.7 --seed 42 --numseqs 50 --input_len 512 --output_len 256 --all
--k 14 --f 3 --max_model_len 2048
```

**Env hygiene** (per reviewer v3 #3):
```
SSD_PROFILE_MESA=0
SSD_PROFILE_MESA_DETAIL=0
SSD_PROFILE_DRAFT (unset)
SSD_PROFILE_TARGET (unset)
SSD_PROFILE (unset)
SSD_TRACE_BUCKET (unset)
SSD_TRACE_SPLIT_K1K2 (unset)
```

## Results (3 reps)

| rep | decode_tps (tok/s) | target_full_step_ms |
|---:|---:|---:|
| 1 | 71.29 | 54.76 |
| 2 | 73.56 | 55.53 |
| 3 | 72.21 | 55.11 |
| **mean** | **72.35** | **55.13** |
| stddev (sample) | 0.93 | 0.31 |
| **CoV** | **±1.3 %** | ±0.6 % |
| range | 71.29 – 73.56 | 54.76 – 55.53 |

## Noise floor implications for gates

The TPS coefficient of variation is **±1.3 %** (1σ, 3 reps). This affects
how confidently Phase 5's decision thresholds can be applied:

| improvement | sigmas | confidence |
|---|---:|---|
| +1 % | 0.8σ | within noise — undetectable from 3-rep means |
| +2 % | 1.5σ | weak signal; could be variance |
| +3 % | 2.3σ | reasonably detectable |
| +5 % | 3.8σ | clearly significant |

**Recommendation**: revise Phase 5 default-on threshold from +2 % to +3 %
unless we increase to 5+ reps per condition. opt-in band (+1-2 %) should
be treated as "experiment inconclusive, suggest more measurement" rather
than a confirmed improvement.

step_time is more stable (±0.6 % CoV) → step_time delta is the more
sensitive lever for diagnostics.

## Profile overhead — measured

Comparison against the earlier PROFILE_MESA=1 breakdown
(`experiments/proxy_async_overlap/breakdown/`, same config except ns=20):

| | clean (PROFILE_MESA=0) | breakdown (PROFILE_MESA=1) | overhead |
|---|---:|---:|---:|
| decode_tps | 72.35 | 70.81 | **−2.1 %** |
| target_full_step_ms | 55.13 | 57.40 | **+4.1 %** |

Confirms reviewer feedback v3 #3: profile-on runs are not valid for TPS
judgment. Profile-on stays the reference for wait-shift / per-status /
graph_post analysis (Phase 4), but never for the +N% gate measurements.

## Phase 5 baseline numbers (fixed)

```
TPS_baseline_clean      = 72.35 tok/s   (mean of 3 reps, ±1.3 % CoV)
step_baseline_clean     = 55.13 ms      (mean of 3 reps, ±0.6 % CoV)
+1 % gate               =  73.07 tok/s  (0.8σ — noise-bound)
+2 % gate               =  73.80 tok/s  (1.5σ)
+3 % gate               =  74.52 tok/s  (2.3σ — recommended floor)
+5 % gate               =  75.97 tok/s  (3.8σ — clear win)
```

## Artifacts

```
phase0_baseline/
  run.sh            — reproducible runner (env hygiene + 3 reps)
  master.log        — outer phase-divider log
  rep_1/run.log     — full bench output
  rep_2/run.log
  rep_3/run.log
  RESULTS.md        — this file
```

No `mesa_profile_*.json` produced (PROFILE_MESA=0).

## Next: Phase 1

AsyncSendRing infrastructure in `ssd/utils/async_helpers/nccl_pack.py`.
No wiring yet; just the class so Phase 2 can use it. See
`docs/mesa/08-proxy-overlap-experiment.md` §3 for the spec.
