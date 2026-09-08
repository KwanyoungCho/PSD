# Full Spec-Bench tree comparison

> Profiler-off, fixed paired questions. TPS and AL are question-level; latencies are verification-step weighted.

| Arm | Q (turns) | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ref_k8_k4_e56_n8m8 | 480 (560) | 63.351 | 4.339 | 0.759 | 0.597 | 5.278 | 0.161 | 3.237 | 69.397 | 62.967 |
| winner_k8_k5_e49_n10m10 | 480 (560) | 67.646 | 4.604 | 0.760 | 0.623 | 5.375 | 0.137 | 3.531 | 68.939 | 62.324 |

## Paired change from reference

| Arm | mean ΔTPS | TPS wins | mean ΔAL | AL wins |
|---|---:|---:|---:|---:|
| winner_k8_k5_e49_n10m10 | 4.295 | 276/479 | 0.265 | 268/479 |

## Per-subtask

### ref_k8_k4_e56_n8m8

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 80 (160) | 65.059 | 4.446 | 0.827 | 5.199 | 3.148 |
| translation | 80 (80) | 67.791 | 4.643 | 0.781 | 5.475 | 3.173 |
| summarization | 80 (80) | 56.354 | 3.879 | 0.609 | 5.377 | 3.257 |
| qa | 80 (80) | 66.513 | 4.541 | 0.827 | 5.169 | 3.306 |
| math_reasoning | 80 (80) | 62.138 | 4.229 | 0.796 | 5.102 | 3.255 |
| rag | 80 (80) | 62.168 | 4.290 | 0.710 | 5.369 | 3.294 |

### winner_k8_k5_e49_n10m10

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 80 (160) | 67.883 | 4.610 | 0.802 | 5.241 | 3.373 |
| translation | 80 (80) | 73.250 | 4.968 | 0.769 | 5.715 | 3.746 |
| summarization | 80 (80) | 58.378 | 4.006 | 0.628 | 5.059 | 3.368 |
| qa | 80 (80) | 70.256 | 4.774 | 0.801 | 5.309 | 3.663 |
| math_reasoning | 80 (80) | 68.992 | 4.660 | 0.841 | 5.331 | 3.406 |
| rag | 80 (80) | 67.002 | 4.598 | 0.716 | 5.557 | 3.597 |
