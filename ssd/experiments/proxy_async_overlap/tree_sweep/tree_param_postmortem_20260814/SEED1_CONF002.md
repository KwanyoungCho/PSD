# Seed 1 confidence 0.01 vs 0.02

> Profiler-off, fixed paired questions. TPS and AL are question-level; latencies are verification-step weighted.

| Arm | Q (turns) | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multiseed_s1_n10m10 | 60 (70) | 56.902 | 3.868 | 0.714 | 0.549 | 4.602 | 0.165 | 3.124 | 69.764 | 61.613 |
| threshold_s1_c002 | 60 (70) | 56.892 | 3.870 | 0.702 | 0.531 | 4.655 | 0.171 | 3.270 | 69.842 | 61.661 |

## Paired change from reference

| Arm | mean ΔTPS | TPS wins | mean ΔAL | AL wins |
|---|---:|---:|---:|---:|
| threshold_s1_c002 | -0.010 | 30/60 | 0.002 | 29/60 |

## Per-subtask

### multiseed_s1_n10m10

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 10 (20) | 54.057 | 3.654 | 0.748 | 4.326 | 2.827 |
| translation | 10 (10) | 55.259 | 3.754 | 0.721 | 4.115 | 3.145 |
| summarization | 10 (10) | 49.597 | 3.416 | 0.584 | 4.596 | 3.254 |
| qa | 10 (10) | 64.322 | 4.315 | 0.803 | 4.800 | 3.335 |
| math_reasoning | 10 (10) | 59.046 | 3.985 | 0.783 | 4.748 | 2.969 |
| rag | 10 (10) | 59.130 | 4.083 | 0.646 | 5.072 | 3.250 |

### threshold_s1_c002

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 10 (20) | 53.113 | 3.615 | 0.727 | 4.151 | 3.114 |
| translation | 10 (10) | 58.497 | 3.947 | 0.751 | 4.559 | 3.175 |
| summarization | 10 (10) | 50.034 | 3.440 | 0.522 | 4.893 | 3.282 |
| qa | 10 (10) | 59.871 | 4.045 | 0.752 | 4.488 | 3.470 |
| math_reasoning | 10 (10) | 64.410 | 4.359 | 0.804 | 5.308 | 3.192 |
| rag | 10 (10) | 55.424 | 3.816 | 0.655 | 4.598 | 3.404 |
