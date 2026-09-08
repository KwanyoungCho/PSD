# Profiler-off triad validation

Same tiny7/seed42/output256 triad with `SSD_PROFILE_DUET=0`. This table checks that the detailed CUDA-event instrumentation did not create the observed target-step gap.

| Metric | Chain | P2 tree only | P1+P2 tree |
|---|---:|---:|---:|
| Target step | 67.281 ± 0.762 | 68.298 ± 0.070 | 71.315 ± 0.110 |
| Target verify | 61.189 ± 0.352 | 61.439 ± 0.033 | 62.342 ± 0.090 |
| Outside verify | 6.092 ± 0.409 | 6.859 ± 0.037 | 8.974 ± 0.199 |

## P2 tree − chain

- `target_step_ms`: +1.018 ms
- `target_verify_ms`: +0.250 ms
- `outside_verify_ms`: +0.768 ms

## Full tree − chain

- `target_step_ms`: +4.035 ms
- `target_verify_ms`: +1.153 ms
- `outside_verify_ms`: +2.882 ms
