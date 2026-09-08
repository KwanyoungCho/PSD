# Post-full N2 search-only comparison

> Profiler-off, fixed paired questions. TPS and AL are question-level; latencies are verification-step weighted.

| Arm | Q (turns) | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| postfull_ref_n10m10 | 60 (70) | 57.351 | 3.901 | 0.725 | 0.555 | 4.564 | 0.170 | 3.178 | 69.765 | 61.619 |
| postfull_n12m10 | 60 (70) | 59.278 | 4.027 | 0.706 | 0.567 | 4.786 | 0.139 | 3.376 | 69.777 | 61.604 |

## Paired change from reference

| Arm | mean ΔTPS | TPS wins | mean ΔAL | AL wins |
|---|---:|---:|---:|---:|
| postfull_n12m10 | 1.927 | 39/60 | 0.127 | 39/60 |

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

### postfull_n12m10

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 10 (20) | 55.254 | 3.746 | 0.759 | 4.240 | 3.247 |
| translation | 10 (10) | 67.449 | 4.557 | 0.774 | 5.297 | 3.477 |
| summarization | 10 (10) | 49.591 | 3.413 | 0.530 | 4.825 | 3.767 |
| qa | 10 (10) | 62.495 | 4.233 | 0.726 | 4.798 | 3.393 |
| math_reasoning | 10 (10) | 68.253 | 4.602 | 0.820 | 5.084 | 3.147 |
| rag | 10 (10) | 52.624 | 3.614 | 0.624 | 4.453 | 3.331 |
