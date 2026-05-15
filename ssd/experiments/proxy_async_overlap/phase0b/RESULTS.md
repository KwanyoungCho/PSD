# Phase 0b results — K1=K2=7 baseline measurement

**Date**: 2026-05-15
**Branch**: feat/mesa-proxy-async-overlap @ ca81819
**Verdict**: 🔴 **STOP** — `unattributed_stall < 0.5 ms` gate triggered.

## Setup

- Hardware: RTX 3090 ×5 (TP=4 target + 1 draft)
- Model: layerskip-llama2-70B (AWQ TP=4 target + TinyLlama-1.1B AWQ TP=1 draft)
- K1=K2=7, dfo=2, pfo=1, exit_layer=56, --temp 0.7 --seed 42 --numseqs 50
  --input_len 512 --output_len 256 (faster probe vs the 512 baseline)
- A = `SSD_PROFILE_MESA_DETAIL=0`; B = `SSD_PROFILE_MESA_DETAIL=1`
- Both runs use the exact baseline command from
  `experiments/paper_baselines/final_experiments/20260513_ours_k1_7_k2_7_*/metadata.txt`,
  with `SSD_PROFILE_DIR` and `SSD_PROFILE_MESA_DETAIL` as the only changes.

## Decision gates

### 1. Probe effect: 🟢 GREEN
- `|A.outer_mean − B.outer_mean| = |0.478 − 0.596| = 0.118 ms`
- < 0.3 ms threshold → DETAIL probe does not perturb outer.

### 2. Stall classification: 🔴 RED
- B inner sum mean = `proxy_compute (0.175) + proxy_pack (0.275) + proxy_send (0.100) = 0.549 ms`
- B outer mean = 0.596 ms
- **unattributed_stall_mean = 0.047 ms** → far below 0.5 ms threshold.
- → No headroom in `proxy_compute_send`. Phase 2/3 cannot meaningfully reduce it further.

## Cross-check vs 2026-05-13 baseline (the run the reviewer's 2.31 ms came from)

The prior run was on `feat/mesa-phase2-hybrid @ 1a8af64`, BEFORE the engine WIP
that was committed in this branch as `43c3f6d` (carry-over instrumentation).
Recomputed from the prior run's `mesa_profile_target_rank0_160652.json`:

| metric (target, median) | 2026-05-13 (1a8af64) | 2026-05-15 (this branch A) | Δ |
|---|---:|---:|---:|
| graph_pre | 28.627 | 28.818 | +0.19 |
| exit_logits | 0.463 | 0.510 | +0.05 |
| **proxy_compute_send** | **2.310** | **0.349** | **−1.96** |
| graph_post | 11.034 | 11.042 | +0.01 |
| verify_sample_accept | 3.714 | 2.633 | −1.08 |
| final_logits | 0.344 | 0.329 | −0.02 |
| **target_spec_wait_*** | **4.207** | **6.908** | **+2.70** |

| TPS-level (run summary) | 2026-05-13 | 2026-05-15 (A) | Δ |
|---|---:|---:|---:|
| target_full_step_ms | 55.96 | 56.94 | +0.98 |
| decode_tps | 74.79 | 72.23 | **−3.4 %** |

## Interpretation

The reviewer's premise — that `proxy_compute_send` ≈ 2.31 ms is a target
critical-path stall worth optimizing — **was correct at the time of the
2026-05-13 baseline.** It is **no longer true** as of the current branch.
Between then and now, the engine WIP carried over in commit `43c3f6d`
(profile_cache_status refactor: replaces per-step `.item()` syncs in
step.py / verifier.py / draft_runner.py with a pre-computed CPU value
threaded through `SpeculateResult.profile_cache_status`) had a side-effect
of shifting where the stream-visible stall lands.

The numerically dominant change is `proxy_compute_send` ↓ by ≈2 ms and
`target_spec_wait` ↑ by ≈2.7 ms. Net `target_full_step` is virtually
unchanged (+0.98 ms); decode TPS is slightly **worse** (−3.4 %).

This is **exactly the "wait shift" anti-pattern** the reviewer warned about
for Phase 2/3 — proxy_outer drops but the equivalent time reappears in
spec_wait, so end-to-end throughput does not improve.

## Conclusion — predefined STOP condition is in force

Per the plan:
> If `unattributed_stall < 0.5 ms` → STOP, no headroom; revisit bottleneck source.

`proxy_compute_send` has ≈0.05 ms unattributed stall. Phase 2 (async send +
ring buffer) and Phase 3 (proxy_stream overlap) would attack a ≈0.5 ms total
span, with **no plausible path to a TPS improvement** — and the wait-shift
mechanism is already attested in the empirical comparison above.

**Do not proceed with Phase 2/3 as originally designed.**

## Where the real headroom is (target side, current run, mean)

```
graph_pre               28.80 ms  (53%)   ← TP forward layers [0..56], fixed CG cost
graph_post              11.05 ms  (20%)   ← layers [57..79] + final norm, fixed CG cost
target_spec_wait_*       9.01 ms  (17%)   ← waiting for draft; p99=19.96 ms (heavy tail)
verify_sample_accept     2.71 ms   (5%)   ← verify() Python work
exit_logits              0.57 ms   (1%)   ← norm + lm_head for proxy
proxy_compute_send       0.48 ms   (1%)   ← the thing we were going to optimize
verify_setup             0.24 ms
final_logits             0.33 ms
─────────────────────
target_full_step ≈     56.94 ms
```

`graph_pre + graph_post` are fundamental forward-pass costs (TP collectives
inside a CUDA graph) and outside the scope of a wire-format optimization.
The realistic optimization targets, in order of expected payoff:

1. **`target_spec_wait` mean 9 ms / p99 19.96 ms** — half of mean is heavy-tail
   p99 outliers. Investigating *which* draft-side step delays produce these
   tails could yield 1–3 ms mean savings (≈2–5 % TPS).
2. **`verify_sample_accept` 2.71 ms** — verify() Python path. Possibly some
   `.item()` syncs or unnecessary CPU work hiding here.
3. (deferred) `proxy_compute_send` — already at 0.48 ms, no headroom.

These are the directions to explore next, NOT the proxy-send overlap.

## Artifacts

```
phase0b/
  A_detail0/mesa_profile_target_rank0_124719.json
  A_detail0/mesa_profile_draft_124722.json
  A_detail0/run.log
  B_detail1/mesa_profile_target_rank0_*.json
  B_detail1/mesa_profile_draft_*.json
  B_detail1/run.log
  analyze_output.txt
  RESULTS.md (this file)
```
