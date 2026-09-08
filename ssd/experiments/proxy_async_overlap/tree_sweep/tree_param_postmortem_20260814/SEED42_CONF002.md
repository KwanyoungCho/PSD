# Seed 42 confidence 0.01 vs 0.02

> Profiler-off, fixed paired questions. TPS and AL are question-level; latencies are verification-step weighted.

| Arm | Q (turns) | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| postfull_ref_n10m10 | 60 (70) | 57.351 | 3.901 | 0.725 | 0.555 | 4.564 | 0.170 | 3.178 | 69.765 | 61.619 |
| threshold_s42_c002 | 60 (70) | 60.570 | 4.125 | 0.720 | 0.559 | 4.941 | 0.161 | 3.401 | 69.940 | 61.608 |

## Paired change from reference

| Arm | mean ΔTPS | TPS wins | mean ΔAL | AL wins |
|---|---:|---:|---:|---:|
| threshold_s42_c002 | 3.220 | 35/60 | 0.225 | 35/60 |

## Per-subtask

### postfull_ref_n10m10

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 10 (20) | 51.232 | 3.467 | 0.746 | 3.927 | 3.081 |
| translation | 10 (10) | 58.549 | 3.973 | 0.676 | 4.687 | 2.737 |
| summarization | 10 (10) | 55.044 | 3.783 | 0.725 | 4.320 | 3.031 |
| qa | 10 (10) | 57.840 | 3.907 | 0.749 | 4.494 | 3.384 |
| math_reasoning | 10 (10) | 55.826 | 3.775 | 0.716 | 4.874 | 3.062 |
| rag | 10 (10) | 65.613 | 4.500 | 0.739 | 5.114 | 3.810 |

### threshold_s42_c002

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 10 (20) | 60.869 | 4.109 | 0.772 | 4.873 | 3.059 |
| translation | 10 (10) | 63.614 | 4.301 | 0.730 | 5.089 | 3.178 |
| summarization | 10 (10) | 65.397 | 4.529 | 0.755 | 5.001 | 3.763 |
| qa | 10 (10) | 60.714 | 4.126 | 0.753 | 4.491 | 3.656 |
| math_reasoning | 10 (10) | 51.305 | 3.473 | 0.624 | 4.846 | 3.317 |
| rag | 10 (10) | 61.524 | 4.214 | 0.684 | 5.424 | 3.465 |
