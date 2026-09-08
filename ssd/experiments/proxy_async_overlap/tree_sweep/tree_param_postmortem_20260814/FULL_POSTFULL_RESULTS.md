# Full Spec-Bench: P2 search cap 10 vs 12

> Profiler-off, fixed paired questions. TPS and AL are question-level; latencies are verification-step weighted.

| Arm | Q (turns) | TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step (ms) | Verify (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| winner_k8_k5_e49_n10m10 | 480 (560) | 67.646 | 4.604 | 0.760 | 0.623 | 5.375 | 0.137 | 3.531 | 68.939 | 62.324 |
| postfull_candidate_n12m10 | 480 (560) | 66.652 | 4.551 | 0.749 | 0.611 | 5.394 | 0.138 | 3.516 | 69.181 | 62.438 |

## Paired change from reference

| Arm | mean ΔTPS | TPS wins | mean ΔAL | AL wins |
|---|---:|---:|---:|---:|
| postfull_candidate_n12m10 | -0.994 | 245/479 | -0.054 | 246/479 |

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

### postfull_candidate_n12m10

| Subtask | Q (turns) | TPS | AL | Hit | P1 AL | P2 AL |
|---|---:|---:|---:|---:|---:|---:|
| mt_bench | 80 (160) | 68.785 | 4.685 | 0.809 | 5.339 | 3.465 |
| translation | 80 (80) | 72.686 | 4.949 | 0.767 | 5.783 | 3.466 |
| summarization | 80 (80) | 54.514 | 3.755 | 0.596 | 4.913 | 3.240 |
| qa | 80 (80) | 70.015 | 4.760 | 0.798 | 5.385 | 3.760 |
| math_reasoning | 80 (80) | 63.327 | 4.297 | 0.776 | 5.162 | 3.479 |
| rag | 80 (80) | 70.434 | 4.847 | 0.748 | 5.686 | 3.624 |
