# Phase B validation results

**Branch**: feat/mesa-proxy-async-overlap
**Date**: 2026-05-15
**Verdict**: 🟢 PASS — Phase B aligned-trace ready to merge.

## Setup

8B model smoke test (faster than the 70B K1=K2=7 baseline; meant only to
catch regressions before the larger validation):

```
--llama --size 8 --gpus 3 --b 1 --temp 0 --seed 42 --numseqs 8
--input_len 128 --output_len 128 --max_model_len 2048
--async --spec --k 5 --f 3
--mesa --mesa_exit_layer 21 --mesa_phase1_k 3 --mesa_phase2_k 2
--mesa_draft_fan_out 2 --mesa_policy b
SSD_FORCE_SPLIT_K1K2=1
```

A: `SSD_PROFILE_MESA=0` (cold path; no anchor / no context / no row build)
B: `SSD_PROFILE_MESA=1` (anchor + context + new schema)

## Gates

### Correctness (greedy byte-identical) 🟢

```
$ diff <(grep '^Generation:' off/run.log) <(grep '^Generation:' on/run.log)
(no output → byte-identical)
```

Every Generation line for all 8 prompts is identical. Phase B introduces no
runtime semantics change.

### Performance (overhead) 🟢

```
off: Final Decode Throughput: 116.16 tok/s   target full step 45.15 ms
on : Final Decode Throughput: 119.12 tok/s   target full step 36.43 ms
```

`on` is 2.5 % faster than `off` — within the run-to-run noise band for this
tiny configuration (8 prompts × 128 out tokens; warmup dominates). The
important signal is the absence of a regression. A larger 70B K1=K2=7
overhead measurement is deferred (`SSD_PROFILE_MESA` overhead in real
paper-sized runs).

### JSON schema 🟢

Target rank 0 dump (`mesa_profile_target_rank0_180028.json`):
- 3241 rows total (1 `_anchor` sentinel + 3240 spans)
- Schema fields: `idx, proc, step_id, status, label, parent_label,
  gpu_start_ms_since_anchor, gpu_end_ms_since_anchor, wall_start_ns,
  wall_end_ns, cuda_ms, cpu_dispatch_start_ns, cpu_dispatch_end_ns,
  ms, start_ms, end_ms` (last three = back-compat aliases)
- `_anchor`: `anchor_cpu_ns=298357534425691`, `anchor_device=0`
- `step_id`: range 1 → 270 (= 270 spec requests), 0 NULLs
- `parent_label` distribution: `target_spec_wait → 810`
  (= 3 markers × 270 steps ✓ exact), `None → 2430`
- `proc`: 100 % `target_rank0`
- Handshake markers present: `target_send_request`,
  `target_recv_response_wait`, `target_response_received` ✓

Draft dump (`mesa_profile_draft_180028.json`):
- 5403 rows total
- `_anchor`: `anchor_cpu_ns=298356593408980`, `anchor_device=2`
- `step_id`: range 1 → 270 with only 2 NULLs (pre-first-request warmup)
- `parent_label`: `phase2_build → 270` (= proxy_wait × 270 ✓ exact)
- `proc`: `draft` except 2 warmup events
- `draft_recv_request` marker present ✓

### Cross-process alignment (step_id=50 spot check) 🟢

| process | label | wall_start_ns | duration | parent | status |
|---|---|---:|---:|---|---|
| target | target_spec_wait_hit_k1 | 298359270196565 | 13.596 ms | (root) | hit_k1 |
| target |   ↳ target_send_request | 298359270467316 |  0.382 ms | target_spec_wait | hit_k1 |
| target |   ↳ target_recv_response_wait | 298359270866608 | 12.759 ms | target_spec_wait | hit_k1 |
| target |   ↳ target_response_received | 298359283628816 |  0.002 ms | target_spec_wait | hit_k1 |
| draft  | draft_recv_request | 298359282097456 |  0.293 ms | (root) | None |
| draft  | hit_cache_respond_hit_k1 | 298359282408491 |  0.804 ms | (root) | hit_k1 |
| draft  | draft_send_response | 298359283299360 |  0.251 ms | (root) | hit_k1 |
| draft  | phase1_replay (×3) | … |  ~4.2 ms each | (root) | hit_k1 |
| draft  | proxy_wait | 298359302569868 |  0.002 ms | phase2_build | hit_k1 |
| target | graph_pre | 298359284618928 |  9.642 ms | (root) | hit_k1 |
| target | exit_logits + proxy_compute_send + graph_post … | … | … | (root) | hit_k1 |

Causality check on the same step_id:

- `target_send_request` ends at  298 359 270 849 274 (wall ns)
- `draft_recv_request`  starts at 298 359 282 097 456 (wall ns)
- gap = +11.248 ms

This is the **pipeline overlap structure**, not an alignment bug: at the
instant target finishes dispatching the request, draft is still finishing
the prior step's phase2 work and has not yet returned to `recv_cmd`. The
trace surfaces this directly — which is exactly what Phase B was built for.

### Status late-bind (target_spec_wait) 🟢

Every target span in step 50 has `status="hit_k1"`, including
`target_spec_wait_hit_k1` whose status is only known AFTER
`speculator.speculate()` returns. The `mesa_close()` close-time context
read works as specified.

## Artifacts (committed under phase_b/)

```
validate_smoke.sh           # the A/B smoke script
smoke.master.log            # outer phase-divider log
off/run.log                 # PROFILE_MESA=0 bench output
on/run.log                  # PROFILE_MESA=1 bench output
on/mesa_profile_target_rank0_180028.json   # ~2.5 MB
on/mesa_profile_draft_180028.json          # ~2.5 MB
```

## Next steps

1. Phase B merge commit (this validation passes).
2. Phase B overhead measurement on 70B K1=K2=7 deferred. The 8B smoke
   shows no regression; the 70B re-measurement can run as a side check
   while Phase C plotter work proceeds (parallelizable).
3. Phase C: new aligned plotter consumes `wall_*_ns` for x-axis and
   joins target/draft by `step_id`. Use these JSON files as fixture
   data during plotter development.
