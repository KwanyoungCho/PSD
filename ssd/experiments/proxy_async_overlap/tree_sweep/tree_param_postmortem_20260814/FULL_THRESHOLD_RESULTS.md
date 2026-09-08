# Full Spec-Bench: P2 confidence 0.01 vs 0.02

> Profiler-off, fixed paired questions. TPS and AL are question-level; latencies are verification-step weighted.

| Arm | Q (turns) | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| winner_k8_k5_e49_n10m10 | 480 (560) | 67.646 | 4.604 | 0.760 | 0.623 | 5.375 | 0.137 | 3.531 | 68.939 | 62.324 |
| postfull_conf002_n10m10 | 480 (560) | 66.219 | 4.517 | 0.743 | 0.605 | 5.344 | 0.138 | 3.524 | 69.151 | 62.381 |

## Paired change from reference

| Arm | mean ΔTPS | TPS wins | mean ΔAL | AL wins |
|---|---:|---:|---:|---:|
| postfull_conf002_n10m10 | -1.427 | 230/479 | -0.087 | 227/479 |

## Per-subtask

### winner_k8_k5_e49_n10m10

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 80 (160) | 67.883 | 4.610 | 0.802 | 5.241 | 3.373 |
| translation | 80 (80) | 73.250 | 4.968 | 0.769 | 5.715 | 3.746 |
| summarization | 80 (80) | 58.378 | 4.006 | 0.628 | 5.059 | 3.368 |
| qa | 80 (80) | 70.256 | 4.774 | 0.801 | 5.309 | 3.663 |
| math_reasoning | 80 (80) | 68.992 | 4.660 | 0.841 | 5.331 | 3.406 |
| rag | 80 (80) | 67.002 | 4.598 | 0.716 | 5.557 | 3.597 |

### postfull_conf002_n10m10

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 80 (160) | 68.018 | 4.622 | 0.804 | 5.300 | 3.410 |
| translation | 80 (80) | 72.580 | 4.937 | 0.775 | 5.597 | 3.513 |
| summarization | 80 (80) | 57.107 | 3.920 | 0.609 | 5.129 | 3.445 |
| qa | 80 (80) | 69.387 | 4.711 | 0.781 | 5.307 | 3.685 |
| math_reasoning | 80 (80) | 64.065 | 4.350 | 0.779 | 5.148 | 3.460 |
| rag | 80 (80) | 66.042 | 4.558 | 0.711 | 5.551 | 3.623 |
